# Plan réaliste sur 2× RTX 3090 (48 GB VRAM total)

*Mai 2026. Réécriture honnête de la roadmap HELIX pour un setup hobby/indépendant : 2× RTX 3090 (24 GB chacune, PCIe gen4, pas de NVLink utile, ~35 TFLOPS BF16).*

---

## Ce qui est impossible

**Soyons clairs avant de planifier :**

| Tâche | Pourquoi impossible | Compute requis |
|---|---|---|
| Distiller un 7B en hybride GDN/attention | Activations long-context OOM, optimizer states ne tiennent pas | 256× H100 × 4 jours = ~25K H100-h |
| Continued pretraining 7B sur 10B tokens long-form | Throughput trop faible, ~250 jours non-stop | 5K H100-h minimum |
| RL sur jetons-curseur (multi-hop, signal sparse) | Besoin de batches massifs, rollouts longs | 10K+ H100-h |
| Phase 2 et 3 de HELIX | Hors d'ordre de grandeur | 50K+ H100-h |

**Ratio brut** : 1× 3090 ≈ 0.15× H100 en BF16, et **0.05×** sur les workloads longs (pas de FP8, mémoire 3× plus petite, pas de NVLink). Pour atteindre les 75 000 H100-h budget initial, **il faudrait 2× 3090 pendant ~85 ans**. Game over.

---

## Ce qui est possible

Avec 2× 3090 (48 GB total), trois projets ont du sens, par ordre de valeur croissante mais aussi de risque croissant :

### Projet A — Stack d'inférence long-context state-of-the-art (3 mois, faisable, fort impact)

**Objectif** : prendre un modèle 1M déjà entraîné (Qwen2.5-7B-1M ou Llama-3.1-8B), construire la stack d'inférence la plus aggressive possible, **dépasser la qualité native du modèle à 1M et étendre par retrieval à ~5M**.

C'est un projet **engineering, pas recherche fondamentale**. Mais c'est un livrable open-source qui n'existe pas aujourd'hui.

**Stack complète :**
```
Qwen2.5-7B-1M (fp16, ~14 GB) sur GPU 0
  └─> vLLM ou SGLang fork
      ├─ PagedAttention (substrat)
      ├─ KVQuant nuq2 (compression KV 8×, 2-bit)
      ├─ Quest top-K page selection (25% actif)
      ├─ StreamingLLM 4 sinks
      ├─ MInference Vertical-Slash (prefill 10× plus rapide)
      └─ Prefix cache automatique

GPU 1 (24 GB) → réservé à :
  ├─ KV cold pages offload (style LayerKV/InfiniGen)
  ├─ EM-LLM episodic memory (kNN retrieval store)
  └─ JEPA encoder/decoder pour compression latente long terme
```

**Calcul mémoire à 1M tokens** :
- Weights : 14 GB
- KV après KVQuant nuq2 : 33 / 8 = **4.1 GB**
- Activations + workspace : ~3 GB
- **Total GPU 0 : ~21 GB → tient sur une 3090 de 24 GB.**

**Calcul à 5M tokens (mode étendu)** :
- KV nuq2 = 20.5 GB → trop pour GPU 0 + weights
- Solution : KV pages chaudes (dernier 1M) sur GPU 0 (4.1 GB), KV pages froides (4M restants, 16 GB) sur GPU 1
- GPU 1 a aussi le JEPA decoder + EM-LLM store
- **Faisable mais latence dégradée** sur les pages froides (PCIe gen4 = ~32 GB/s)

**Cible mesurable** :
| Métrique | Seuil de succès |
|---|---|
| RULER@1M | ≥ Qwen2.5-7B-1M baseline (~80) |
| RULER@5M (avec retrieval EM-LLM) | ≥ 50 |
| Throughput décodage @ 1M | ≥ 5 tok/s sur 1× 3090 |
| TTFT @ 1M (prefill) | ≤ 60 s |
| Mémoire pic @ 5M | ≤ 22 GB GPU 0 + 22 GB GPU 1 |

**Effort** : 1 personne, 3 mois temps plein. **Coût compute : ~0$** (inférence sur modèle déjà entraîné, juste du dev/test).

**Livrable** : un fork open-source de vLLM ou SGLang avec doc et benchmarks. Vraie valeur ajoutée pour la communauté ; aujourd'hui personne n'a une stack qui combine ces 5 composants proprement.

---

### Projet B — Prototype d'un seul composant novateur sur petit modèle (4–6 mois, recherche)

Une fois Projet A en route, choisir **un seul** des composants spéculatifs de HELIX et le valider à petite échelle. Le but n'est pas de battre le SOTA — c'est d'apprendre si l'idée marche **du tout**.

#### Option B1 — Encoder JEPA pour compression contextuelle

**Setup** :
- Modèle base : **Qwen2.5-1.5B** (3.1 GB en fp16, fits one 3090 with massive headroom)
- Train le JEPA encoder/decoder sur des chunks de 2048 tokens → 64 latents
- Objectif : reconstruction des embeddings du tronc gelé (couches 8–16)
- Données : ~1B tokens de long-form (PG-19, code, articles)
- QLoRA pour entraîner uniquement l'encoder/decoder

**Mémoire** : Qwen2.5-1.5B + activations 32K + JEPA encoder 4 couches 256d ~1.5 GB. Tout tient sur 1× 3090.

**Compute** : ~2000 3090-hours (~40 jours non-stop sur 2× 3090) pour 1B tokens à 32K context.

**Falsifiabilité** : si après 1B tokens, le decoder ne reconstruit pas les embeddings cibles à >85% cosine similarity, l'objectif latent ne marche pas → essayer objectif token-level (LCC-style).

**Livrable** : papier ou blog post sur "JEPA pour compression de contexte fonctionne / ne fonctionne pas".

#### Option B2 — Cursor tokens via gating différentiable

C'est la **reformulation prudente** des jetons-curseur de HELIX. Au lieu d'actions discrètes émises dans le vocab, c'est un **gate continu par chunk** qui sélectionne quoi décompresser :

```python
# pseudocode
chunk_scores = MLP(query_embedding, chunk_embeddings)  # [num_chunks]
top_k_indices = topk(chunk_scores, k=K)
decompressed_kv = jepa_decoder(latents[top_k_indices])
attention(query, concat(local_kv, decompressed_kv))
```

C'est **différentiable end-to-end**, contrairement aux jetons-curseur. Train via la perte habituelle next-token. Beaucoup plus tractable.

**Setup** : même que B1 mais sur Phi-3.5-mini (3.8B) avec QLoRA.

**Compute** : ~3000 3090-hours sur ~6 semaines à 2× 3090.

**Falsifiabilité** : si le gate ne se concentre pas (entropie reste >50% du max après convergence), il n'apprend rien d'utile → revenir à top-k statique.

**Livrable** : code + benchmark sur multi-hop QA à 128K.

#### Option B3 — Hybrid block replacement à petite échelle

**Setup** :
- Modèle base : **Qwen2.5-1.5B** ou **Phi-3.5-mini**
- Remplacer 50% des couches d'attention par GDN via une mini-MOHAWK
- Distiller sur 500M tokens
- Tester l'extrapolation long-context

**Compute** : ~2500 3090-hours (~50 jours).

**Falsifiabilité** : si après 500M tokens, le hybride régresse de >5pt sur MMLU vs base, le ratio 50:50 est mauvais à cette échelle.

**Livrable** : code de conversion + benchmarks à petites échelles. Permet de **valider le ratio GDN/attention** avant de scaler.

---

### Projet C — Recherche pure jouet, contribution conceptuelle (12+ mois)

Si tu veux pousser la frontière conceptuelle et pas seulement engineering, prendre un **modèle minuscule** (50M–125M params, GPT-2 small / GPT-2 medium scale) et y implémenter **toutes les composantes de HELIX** :

- Backbone hybride 75:25
- Pyramide KV à 4 niveaux
- Cursor tokens (gating différentiable)
- TTT-E2E par épisode
- JEPA encoder

**Pourquoi c'est valuable** : un modèle jouet où **tout HELIX coexiste** est une preuve de concept conceptuelle. Si tu publies "voici HELIX-tiny qui dépasse un GPT-2 baseline à long context sur BABILong-1M", c'est une contribution scientifique légitime, **citée par les vrais labs** quand ils auront les moyens de scaler.

C'est l'approche **ARMT** : Rodkin et al. ont démontré 50M sur BABILong avec GPT-2 — exactement le pattern.

**Compute** : ~5000 3090-hours sur 6 mois (modèle 100M ne sature pas même à 32K context).

**Livrable** : papier soumissible NeurIPS/ICLR/ICML.

---

## Recommandation

**Ordre que je conseille :**

1. **Projet A en premier (3 mois)** — engineering inférence sur Qwen2.5-7B-1M.
   - Pas risqué.
   - Livre une vraie valeur tout de suite.
   - Donne une base pour les projets suivants (la stack d'inférence est réutilisable).

2. **Projet B2 ou B1 ensuite (4–6 mois)** — prototyper UN seul composant novateur.
   - B2 (gating différentiable) si tu veux quelque chose qui peut s'intégrer à Projet A.
   - B1 (JEPA encoder) si tu veux quelque chose plus orienté recherche.

3. **Projet C en parallèle (long terme, low priority)** — version jouet de HELIX complet sur petit modèle. Sert de "preuve de concept" et de **publication potentielle**.

**Ne pas faire** : tenter Phase 1 de HELIX original. C'est mathématiquement infaisable.

---

## Optimisations spécifiques 2× 3090

Quelques détails techniques qui changent la vie :

### Mémoire
- **Toujours utiliser 4-bit weights** (GPTQ, AWQ, ou bitsandbytes nf4) pour les modèles 7B+. Réduit 14 GB → 4 GB et libère de l'espace pour activations long-context.
- **Flash Attention 2** est obligatoire (Flash Attention 3 ne supporte pas Ampere). Réduit massivement les pics d'activations.
- **Gradient checkpointing** systématique pour tout fine-tuning à >4K context.
- **DeepSpeed ZeRO Stage 2 ou 3** si tu fais du fine-tuning multi-GPU (mais throughput dégradé par PCIe gen4).

### Compute
- 2× 3090 sans NVLink = communication par PCIe gen4 (~32 GB/s) = **tensor parallelism très lent**. Privilégier **pipeline parallelism** (couches sur GPU 0, autres sur GPU 1) ou **modèle séparé par GPU** (modèle principal sur 0, encoder/decoder JEPA + EM-LLM sur 1).
- Pas de FP8/FP4 native sur Ampere. BF16 est la limite. Cela coûte ~2× vs H100 sur les même workloads.

### Données
- Tokenizer cache (Tokenizers en Rust, pas Python) sinon CPU bottleneck à 32K+ context.
- Datasets longs en pré-tokenized streaming (HF datasets `with_format("torch")` + `streaming=True`).
- Pour Projet A, **ne pas re-tokenizer** les benchmarks — les exécuter de manière persistante.

### Refroidissement
- 2× 3090 = ~700 W sous charge soutenue. PSU 1500 W minimum.
- Travail soutenu de plusieurs jours = boîtier ouvert ou rack.
- Surveiller `nvidia-smi -q -d TEMPERATURE` — throttling à 83°C, kill-switch à 93°C.

---

## Si tu veux scaler plus tard

Quelques options abordables si Projet A ou B donne des résultats prometteurs :

| Option | Coût | Quand |
|---|---|---|
| Lambda Labs A100 80GB (1× spot) | ~$1.10/h ≈ $800/mois | Pour un sprint de continued-pretraining |
| RunPod 4× A100 40GB | ~$5/h ≈ $3600/mois | Pour distillation hybride 7B |
| Together.ai / Modal credits | Variable | Beaucoup de programmes pour chercheurs indépendants |
| Programme NCSA / open compute | Gratuit (avec proposal) | Pour un Projet C papier |
| HuggingFace + AWS research credits | Gratuit avec acceptation | Si tu publies un open-source utile |

Le passage de 2× 3090 (48 GB) à un 1× A100 80GB représente ~3× la VRAM et ~2× la compute, ouvrant tout ce qui demande des activations long-context (32K+) en fine-tuning.

---

## Conclusion brutale mais utile

**Le HELIX original est mort sur 2× 3090.** Ce qui survit :

1. La **vision architecturale** (hybride sub-quadratique + pyramide KV + retrieval driven by reasoning) reste valable comme cadre conceptuel.
2. Le **stack d'inférence** (Projet A) est immédiatement constructible et valuable.
3. **Un seul composant novateur** peut être validé scientifiquement sur petit modèle (Projet B).
4. **Tout HELIX en miniature** (Projet C) est une publication potentielle.

**Ne pas faire** : promettre 5M-10M-50M context. Ce n'est pas là que la valeur est sur ce setup.

**Faire** : un Qwen2.5-7B-1M servi avec qualité state-of-the-art à 1M effectif, plus retrieval EM-LLM à 5M de "context périphérique", plus une preuve de concept pour **un** composant novateur — c'est un projet de 9 mois solo qui a un vrai impact open-source.
