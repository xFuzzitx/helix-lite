# HELIX — Architecture pour Contexte Extrême (5M / 10M / 50M tokens)

*Hierarchical Episodic Linear-attention with Indexed eXtension*

*Conception : mai 2026. Cible : transformer un modèle 7B existant (RoPE, GQA) en système HELIX capable de 5M effectifs (phase 1), 10M (phase 2), 50M (phase 3).*

---

## Résumé exécutif

L'état de l'art 2026 (MiniMax-Text-01 à 0.91 RULER@1M, Llama 4 à 10M annoncé/effectif ~5–6.5M, DeepSeek V4 avec MRCR 0.59@1M) **converge sur 5 idées** : cœur sous-quadratique, attention dense fractionnaire comme ancre, compression KV agressive, curriculum progressif, indexeur-puis-attention. **Aucun système open-source ne dépasse 80% de RULER à 5M.** Au-delà de 10M, seul ARMT a démontré 80% sur BABILong à 50M, en architecture jouet (GPT-2).

HELIX combine ce socle convergent avec **5 innovations spécifiques** que la littérature 2025–2026 commence à effleurer mais n'a pas réunies :

1. **Pyramide KV multi-résolution avec compression réversible** (extension de R3Mem et PyramidKV vers un spectre continu, pas des niveaux discrets).
2. **Jetons-curseur programmables** (`⟨ZOOM-IN⟩`, `⟨ZOOM-OUT⟩`, `⟨GOTO ep_k⟩`) qui rendent la rétention contextuelle pilotée par la chaîne de raisonnement, pas par l'attention implicite. **Pas de prior art identifié au format jeton-natif** — MemGPT fait pareil via tool-calls externes.
3. **État dual rapide/lent dans le tronc SSM** avec checkpoints aux frontières d'épisodes EM-LLM, permettant un retour en arrière O(1) sans réattention globale.
4. **Compression auto-distillée à objectif JEPA-style** : chaque niveau de la pyramide apprend à prédire la représentation latente du niveau inférieur, pas les tokens bruts. Étend Latent Context Compilation (arXiv 2602.21221) à un objectif latent prédictif récursif.
5. **End-to-End Test-Time Training (TTT-E2E)** sur les couches d'attention conservées : la mémoire neuronale s'adapte par descente de gradient pendant l'inférence, validé à 35× speedup sur 2M (NVIDIA, 2026).

**Plan en 3 phases :**
- **Phase 1 (4 mois, ~25K H100-h)** : 7B existant → HELIX-5M. Distillation hybride 75% Gated DeltaNet + 25% sparse-softmax + pyramide à 4 niveaux + jetons-curseur. Cible : >75% RULER@1M, >55% à 5M.
- **Phase 2 (8 mois, +30K H100-h)** : ajout TTT-E2E + élargissement de la pyramide. Cible : >80% RULER@1M, >65% à 5M, >50% à 10M.
- **Phase 3 (12 mois, +20K H100-h)** : intégration ARMT-style segment recurrence + JEPA-compression. Cible : >55% à 10M, >40% à 50M sur BABILong.

---

## 1. Le constat — ce qu'aucun système 2026 ne fait

| Limite SOTA | Cause technique | Conséquence à 5M+ |
|---|---|---|
| MiniMax-Text-01 à 0.91 RULER@1M chute à <0.80 estimé@4M | Lightning Attention sans ancre suffisante de récupération | Recall multi-hop s'effondre |
| Llama 4 Scout claim 10M, eff. ~5–6.5M | iRoPE + chunked attn n'ont **jamais vu** des patterns à 1M+ en training | Le modèle "voit" mais ne raisonne pas |
| DeepSeek V3.2 DSA top-2048 à O(L·k) | k=2048 fixe ; pas de raisonnement adaptatif sur quoi récupérer | Retrieval = scoring statique |
| EM-LLM 10M retrieval validé | Externe (kNN sur store), pas intégré au backbone | Pas d'attention dense possible sur les épisodes |
| ARMT 50M sur BABILong | GPT-2 base, BPTT through segments, pas scalable au 7B+ moderne | Inutilisable en production telle quelle |
| Infini-attention 1M passkey | **HF replication failed** — gating ne converge pas | À éviter |
| Titans (Google, 2025) 2M+ NIAH | Code jamais publié, reproductibilité faible (arXiv 2510.09551) | Composant intéressant, pas système |
| TTT-E2E (NVIDIA 2026) 35× @ 2M | Test-time training valide la direction | **Pas encore intégré à un système hiérarchique** |
| SubQ (annoncé 2026) 12M via SSA | Sub-quadratic sparse attention pure | Pas de mémoire compressée hiérarchique |
| R3Mem (2025) compression réversible | Niveau document/entité, deux niveaux | Pas de spectre continu, pas d'intégration au tronc |

**Le gap :** aucun système n'unit (a) backbone hybride sous-quadratique, (b) pyramide KV avec compression réversible continue, (c) récupération pilotée par raisonnement (jetons-curseur), (d) test-time training, (e) mémoire épisodique avec checkpoints SSM. Ces cinq éléments, pris ensemble, sont la cible de HELIX.

---

## 2. Principes de design

1. **Tout coût scale en O(N) ou O(N·log N), jamais O(N²).** Vérification : à 50M tokens, O(N²) = 2.5×10¹⁵ FLOPs (impossible) ; O(N·k=2048) = 1×10¹¹ (faisable) ; O(N·log N) = 1.3×10⁹ (trivial).

2. **La précision suit la pertinence et la récence.** Aucun token n'est "perdu" mais beaucoup sont compressés. Le réseau doit **pouvoir** récupérer la précision quand il le décide.

3. **Le raisonnement contrôle la mémoire.** Plutôt que des heuristiques d'attention statique (importance, top-k), le modèle émet des jetons de contrôle qui reshape l'attention dynamiquement.

4. **Apprendre à compresser en prédisant des latents, pas des tokens.** Objectif JEPA-style : chaque niveau de compression apprend à reconstruire les **embeddings cibles** du niveau inférieur (pas le texte). C'est plus stable et plus dense en information utile.

5. **Distiller, ne jamais pré-entraîner from scratch.** MOHAWK, Mamba-in-the-Llama, Llamba ont prouvé qu'on convertit un transformer dense en hybride avec <1% des données de pré-entraînement.

6. **Test au-delà de la perplexité.** Évaluation = RULER + MRCR-V2 8-needle + BABILong + NoLiMa, à plusieurs longueurs. Aucun checkpoint promu sans franchir des seuils.

---

## 3. Architecture HELIX

### Schéma d'ensemble

```
                                 Input tokens (jusqu'à 50M)
                                          │
                                          ▼
                      ┌────────────────────────────────────────┐
                      │  Tokenizer + Position (RoPE+iRoPE mix) │
                      └────────────────────────────────────────┘
                                          │
                                          ▼
                      ┌────────────────────────────────────────┐
                      │           BACKBONE HÉLICOÏDAL          │
                      │  32 couches alternant :                │
                      │   • 24× Gated DeltaNet (état dual r/l) │
                      │   • 8× Sparse-Softmax+MLA (NSA-style)  │
                      │  Ratio 3:1, ancres aux couches 4,8,…,32│
                      └────────────────────────────────────────┘
                                          │
                          ┌───────────────┼───────────────┐
                          ▼               ▼               ▼
          ┌────────────────────┐  ┌────────────────┐  ┌──────────────┐
          │  PYRAMIDE KV       │  │ INDEX SÉMANT.  │  │ MÉMOIRE TTT  │
          │ N0 fp16  : 32K     │  │ Lightning Idx  │  │ MLP appris   │
          │ N1 4-bit : 256K    │  │ + EM boundaries│  │ on-the-fly   │
          │ N2 2-bit : 4M      │  │ Bayesian surp. │  │ (Titans-like │
          │ N3 latent: 50M     │  │                │  │  surprise)   │
          │  (auto-encoded)    │  │                │  │              │
          └────────────────────┘  └────────────────┘  └──────────────┘
                          │               │               │
                          └───────────────┼───────────────┘
                                          ▼
                      ┌────────────────────────────────────────┐
                      │  CONTRÔLEUR DE RÉCUPÉRATION (RC-HEAD)  │
                      │  Lit jetons-curseur :                  │
                      │    ⟨ZOOM-IN k⟩ ⟨ZOOM-OUT⟩ ⟨GOTO ep_k⟩│
                      │  Reshape attention masks dynamiquement │
                      └────────────────────────────────────────┘
                                          │
                                          ▼
                      ┌────────────────────────────────────────┐
                      │            Output / next token         │
                      └────────────────────────────────────────┘
```

### 3.1 Backbone hélicoïdal

**32 couches** (équivalent à un 7B compact), alternance **3:1 GDN:Sparse-Softmax** (suit Qwen3-Next 75:25, validé en production à 1M).

- **Couches GDN (Gated DeltaNet, arXiv:2412.06464)** : état récurrent de dim `d_state=128`, gating delta-rule. O(N) en train et inférence. **État dual** : un état rapide `h_fast` mis à jour par token, un état lent `h_slow` checkpointé aux frontières d'épisodes (Bayesian surprise > seuil) — permet de rejouer une partie du contexte localement sans re-streamer 50M tokens.
- **Couches Sparse-Softmax** : MLA (Multi-Head Latent Attention, DeepSeek-V3) avec `d_c=512` (vs 14k pour MHA), absorption des poids → 12× moins de KV vs GQA. Sparse pattern style **NSA** (compressed/selected/sliding) avec sliding window de 4096 et top-k blocks adaptatif (k=512 par défaut).
- **iRoPE** : sur les 8 couches d'attention, **2 sont NoPE** (pas de RoPE → meilleure extrapolation longue distance, suit Llama 4 Scout). Les 6 autres ont RoPE base 10M. Sur les couches GDN, pas de positional encoding explicite (état récurrent encode l'ordre).

### 3.2 Pyramide KV multi-résolution

**4 niveaux par couche d'attention conservée**, basée sur l'âge et la pertinence :

| Niveau | Précision | Capacité | Source | Fonction |
|---|---|---|---|---|
| **N0** | fp16 GQA | 32K récents | direct | Détail token |
| **N1** | 4-bit (KIVI/KVQuant) | 256K | rétrogradation | Mémoire courte |
| **N2** | 2-bit nuq2 (KVQuant) | 4M | rétrogradation | Mémoire longue |
| **N3** | latents JEPA appris | 50M | auto-encodé | Mémoire épisodique |

**Promotion / rétrogradation** :
- Tokens nouveaux entrent en N0.
- Quand N0 dépasse 32K, les plus anciens sont **rétrogradés** en N1 (passe par un quantizer KIVI).
- Cascade similaire N1→N2.
- N2→N3 : **les tokens d'un épisode entier** (segmenté via Bayesian surprise EM-LLM) sont passés à un encoder JEPA appris, qui produit un nombre fixe de latents (e.g. 64 latents par épisode, indépendamment de la longueur).
- **Rappel ascendant** : lorsque le RC-Head ou l'indexeur sélectionne un latent N3, on **décode** le latent vers N2 via le décodeur JEPA appris (compression réversible, suit R3Mem mais à objectif latent).

**Volumétrie 50M tokens, 7B avec MLA** :
- N0 : 32K × 32 layers × MLA(512+128) bytes ≈ 660 MB
- N1 : 256K × 32 × 640 × 0.5 ≈ 2.6 GB
- N2 : 4M × 32 × 640 × 0.25 ≈ 20.5 GB
- N3 : 50M / chunk_size(2048) × 64 latents × 32 × 128 fp16 ≈ 50M × ~2 KB = 100 MB par layer × 32 = 3.2 GB

**Total : ~27 GB sur GPU + N3 sur CPU si besoin.** Sur H100 (80 GB), il reste 50 GB pour weights + activations.

### 3.3 Index sémantique

Trois sources de scoring fusionnées par un MLP léger (2 couches, 16M params) :

1. **Lightning Indexer** (style DeepSeek V3.2 DSA) : `I_{t,s} = Σ w · ReLU(q_t · k_s)` calculé sur les KV compressés MLA. O(L·k) linéaire.
2. **Frontières d'épisodes** (style EM-LLM) : segmentation du flux par surprise bayésienne ; chaque épisode a un embedding-prototype calculé comme moyenne pondérée des latents N3.
3. **Récence + fréquence d'accès** : tracker LRU pondéré, mis à jour par les jetons-curseur émis.

Le score de chaque chunk/épisode/latent est utilisé par :
- les couches Sparse-Softmax (pour choisir les top-k blocks à attendre densément),
- le RC-Head (pour résoudre `⟨GOTO ep_k⟩`).

### 3.4 Mémoire TTT (Test-Time Training)

Suit **TTT-E2E** (NVIDIA, mai 2026) et la composante neural-memory de **Titans** (sans le reste de Titans, dont la reproductibilité est faible).

- Une **MLP** de petite taille (`hidden=2048`, ~8M params) par groupe de 4 couches.
- Mise à jour **par descente de gradient** pendant l'inférence sur un objectif de **surprise auto-supervisée** (prédiction du prochain état caché de la couche).
- Lecture par les couches GDN comme un input additionnel (style Memory-as-Context de Titans).

**Pourquoi c'est nouveau dans HELIX :** TTT-E2E seul est plat (un MLP, un état). Ici la TTT-MLP est **conditionnée** sur l'épisode courant (embedding-prototype), donc elle apprend des micro-modèles différents par épisode et peut switcher quand le RC-Head fait `⟨GOTO ep_k⟩`.

### 3.5 Contrôleur de récupération (RC-Head)

Une nouvelle tête de sortie qui partage le tronc avec la prediction-head principale. Elle prédit, à chaque pas, soit un token normal, soit un **jeton-curseur**. Trois jetons spéciaux ajoutés au vocabulaire :

| Jeton | Sémantique | Effet sur la pyramide |
|---|---|---|
| `⟨ZOOM-IN [span_id, k]⟩` | "j'ai besoin de plus de détail sur ce span" | Le span désigné est **promu** N2→N1 ou N1→N0 (décodage JEPA / déquantization). Coût : O(taille_span). |
| `⟨ZOOM-OUT⟩` | "le détail de ce span n'est plus utile" | Force la rétrogradation immédiate. |
| `⟨GOTO ep_k⟩` | "saute à l'épisode k" | Le RC-Head charge l'embedding-prototype, restaure le `h_slow` checkpointé de l'épisode, ré-active la TTT-MLP correspondante. **O(1).** |

Pendant l'entraînement, ces jetons sont **supervisés synthétiquement** : on génère des chaînes de raisonnement multi-hop sur des contextes long, et l'oracle place les jetons-curseur aux positions optimales (RL-from-AI-feedback ou supervision directe).

**Ce qui est nouveau ici** : la littérature a vu des opérations de retrieval externes (MemGPT, ToolFormer) mais pas comme **primitive d'attention native**, intégrée au vocabulaire et co-entraînée avec la prédiction de tokens. C'est la contribution la plus risquée et la plus distinctive de HELIX.

### 3.6 Encoder/Decoder JEPA pour N3

- **Encoder** : transformer 4 couches, 256 dim, ingère un chunk de 2048 tokens, produit 64 latents.
- **Decoder** : transformer 4 couches, reconstruit non pas le texte mais les **embeddings** que produirait le tronc principal sur ces tokens (objectif **latent**, suit V-JEPA).
- **Objectif d'entraînement** : `L = ||h_target - decoder(encoder(chunk))||²` où `h_target` est l'embedding produit par les couches 8–16 du tronc gelé.

**Pourquoi prédire des embeddings, pas des tokens ?** Parce qu'un objectif token-level force le decoder à reconstruire des informations syntaxiques inutiles (ponctuation, tokens fonctionnels). L'objectif latent force l'encoder à conserver l'**information sémantique exploitable par le tronc**, ce qui est la quantité réellement utile en aval. C'est l'idée centrale de I-JEPA / V-JEPA appliquée à la compression de contexte (généralisation de Latent Context Compilation, arXiv:2602.21221).

---

## 4. Innovations clés vs prior art

| Composant HELIX | État de l'art proche | Différence |
|---|---|---|
| Pyramide à 4 niveaux avec spectre continu de précision | PyramidKV (2 niveaux), R3Mem (2 niveaux document/entité) | Cascade automatique, latents JEPA en N3, intégrée au tronc, pas un module externe |
| Jetons-curseur natifs `⟨ZOOM-IN/OUT/GOTO⟩` | MemGPT (tool-calls externes) | **Primitives d'attention natives, partie du vocabulaire**, supervision synthétique de leur émission. Pas de prior art identifié au format jeton-natif. |
| État dual rapide/lent SSM avec checkpoints aux épisodes | Mamba/GDN (état unique), EM-LLM (boundaries externes) | Couplage direct surprise-bayésienne ↔ checkpoint SSM ↔ TTT-MLP conditionnée |
| Compression auto-distillée à objectif latent JEPA | Latent Context Compilation (arXiv:2602.21221, Buffer Tokens via LoRA jetable) | LCC : un seul niveau, supervision KL token-level. HELIX : récursif, latent-level, cascade |
| TTT-E2E conditionné par épisode | TTT-E2E (NVIDIA, 2026) | NVIDIA : un seul MLP global. HELIX : MLP par groupe, conditionné sur embedding-prototype d'épisode |

**En réunissant ces 5 contributions**, HELIX adresse simultanément les 3 défaillances du SOTA :
- **Quality drop multi-hop à 5M+** → couches Sparse-Softmax + RC-Head guidé par raisonnement.
- **KV memory wall à 50M** → pyramide à 4 niveaux (27 GB pour 50M sur 7B-MLA).
- **Patterns d'attention jamais vus en train** → curriculum 32K→1M→10M→50M avec PoSE et compression-aware augmentation, plus jetons-curseur supervisés synthétiquement.

---

## 5. Phase 1 — 7B → HELIX-5M (4 mois, ~25 000 H100-h)

### Choix du modèle de base

**Qwen2.5-7B-1M** ou **Llama-3.1-8B-Instruct**. Qwen est préférable car son curriculum de 5 étapes RoPE-base 10K→10M est déjà fait et son MInference Vertical-Slash sparse pattern est intégrable directement dans le slot Sparse-Softmax.

### Étape 1.1 — Distillation hybride (mois 1, ~10 000 H100-h)

- Partir de Qwen2.5-7B-1M dense (28 couches d'attention dans l'original).
- **Remplacer 21 couches d'attention par 21 couches Gated DeltaNet** (suit Mamba-in-the-Llama, arXiv:2408.15237 ; ratio 75:25).
- Conserver 7 couches d'attention — celles aux indices `4, 8, 12, 16, 20, 24, 28` — converties en MLA (`d_c=512`) avec NSA-style sparse pattern.
- Ajouter le tronc dual-state aux GDN (mod léger sur leur recurrence).
- Distillation : ~10 milliards de tokens, objectif KL token-level (recipe Llamba/MOHAWK), 256 H100s × 4 jours.
- Gel : tronc + couches d'attention. Entraîner uniquement les nouvelles couches GDN et la projection MLA.

### Étape 1.2 — Pyramide KV (mois 2, ~3 000 H100-h)

- Implémenter les 4 niveaux côté inference engine (vLLM-fork ou SGLang-fork).
- Entraîner l'encoder/decoder JEPA pour N3 :
  - Tronc gelé fournit les embeddings cibles.
  - 5B tokens long-form (livres, repos de code de >1M tokens, articles scientifiques).
  - 32 H100s × 5 jours.
- **Pas de retraining du tronc à cette étape** — la pyramide est un wrapper d'inférence.
- Tests : RULER@1M doit rester ≥ ce qu'il était sur le base model post-distillation.

### Étape 1.3 — Jetons-curseur (mois 3, ~7 000 H100-h)

- **Génération de données synthétiques** : prendre 50K documents ≥ 100K tokens. Générer des questions multi-hop nécessitant info de 3 à 7 endroits différents. Pour chaque question, produire une chaîne de pensée `(token, ⟨ZOOM-IN span_i⟩, token, ⟨ZOOM-IN span_j⟩, ..., réponse)` avec les span_i optimaux placés par un oracle (ou GPT-4-class teacher).
- Ajouter les 3 jetons spéciaux au vocabulaire (3 tokens en plus, ~minuscule).
- Fine-tuning supervisé du RC-Head et léger fine-tuning du tronc (LoRA r=64) sur ce dataset.
- 128 H100s × 5 jours.
- Évaluer sur RULER multi-hop, MRCR-V2, BABILong-1M, BABILong-5M.

### Étape 1.4 — Curriculum de longueur (mois 4, ~5 000 H100-h)

- Continued pretraining progressif : 32K → 256K → 1M → 5M.
- À chaque palier : LongLoRA S²-Attn pour réduire le coût, PoSE pour échantillonner les offsets jusqu'à L_target, RoPE base 500M (LongRoPE2 evolutionary search initial sur les couches d'attention).
- Données : ~10B tokens long-form mixés avec 20% de données courtes pour préserver la qualité @ 32K.
- 256 H100s × 4 jours par palier × 4 paliers.

### Cible Phase 1

| Métrique | Seuil de promotion |
|---|---|
| RULER avg @ 32K | ≥ 92 (ne pas régresser vs base) |
| RULER avg @ 1M | ≥ 75 (Qwen2.5-1M était à >80 sur passkey ; la conversion hybride induit une régression ~5-10pt qu'on doit récupérer) |
| RULER avg @ 5M | ≥ 55 |
| MRCR-V2 8-needle @ 1M | ≥ 25 (Sonnet 4.5 : 21, Gemini 2.5 Pro : 16.4) |
| BABILong @ 5M | ≥ 50% |
| Throughput décodage @ 1M | ≥ 4× Qwen2.5-7B dense (la pyramide doit rapporter, pas coûter) |
| KV memory @ 5M | ≤ 35 GB (tient sur H100 80GB avec headroom) |

---

## 6. Phase 2 — HELIX-10M (8 mois, +30 000 H100-h)

### Étape 2.1 — Intégration TTT-E2E

- Ajouter une MLP TTT (`hidden=2048`) par groupe de 4 couches → 8 MLPs au total dans le 7B.
- Mécanisme de mise à jour : à chaque token, descente de gradient sur surprise auto-supervisée (objectif TTT-E2E, suit la recipe NVIDIA).
- Conditionner chaque MLP sur l'embedding-prototype de l'épisode courant (non plus une MLP unique partagée).
- Re-training du tronc complet (LoRA r=128) sur 5B tokens long-form pour intégrer la lecture du TTT-state.
- 256 H100s × 10 jours.

### Étape 2.2 — Élargissement de la pyramide

- N3 capacité étendue à 50M (déjà conçu pour, juste augmenter le budget GPU+CPU).
- Améliorer le decoder JEPA (4 couches → 8 couches) pour réduire la perte de promotion N3→N2.
- Ajouter une **5ème niveau N4 latent global** : un seul vecteur par épisode résumant le tout, utilisé pour le routing GOTO.

### Étape 2.3 — Curriculum 5M → 10M

- Synthèse de données 5M–10M : concaténation de livres entiers, tâches sur monorepos de code de plusieurs millions de tokens, transcripts de longues conversations multi-modales.
- 256 H100s × 12 jours.

### Cible Phase 2

| Métrique | Seuil |
|---|---|
| RULER @ 1M | ≥ 80 |
| RULER @ 5M | ≥ 65 |
| RULER @ 10M | ≥ 50 |
| MRCR-V2 8-needle @ 1M | ≥ 35 |
| BABILong @ 10M | ≥ 40% |
| Latence décodage @ 10M | constante en N (atout TTT-E2E) |

---

## 7. Phase 3 — HELIX-50M (12 mois, +20 000 H100-h)

À ce point, le bottleneck n'est plus l'architecture mais (a) la disponibilité de données utiles à 50M, (b) la stabilité du training à BPTT segments très longs.

### Étape 3.1 — ARMT-style segment recurrence

- Adapter ARMT (arXiv:2407.04841) au tronc HELIX : ajouter une **mémoire associative Hopfield** entre les checkpoints `h_slow` des couches GDN.
- BPTT à travers ~50–100 segments de 1M tokens chacun (truncated BPTT pour stabilité).
- ARMT a démontré 80% sur BABILong à 50M en GPT-2 ; reste à montrer que cela tient à 7B.

### Étape 3.2 — Distillation JEPA récursive

- Ajouter un **niveau N5** : latents méta-épisodes, qui résument 100 épisodes en 64 latents.
- Decoder cascadé : N5→N4→N3→N2→N1→N0 selon les jetons-curseur.

### Étape 3.3 — Données synthétiques 50M

- Le challenge réel : où trouve-t-on des contextes de 50M tokens utiles ? Les options réalistes :
  - **Codebases entiers** (Linux kernel ~30M tokens, Chromium ~50M).
  - **Tous les emails d'une décennie** d'un utilisateur.
  - **Toutes les transcripts vidéo** d'un domaine (e.g. tous les talks NeurIPS depuis 2010).
  - Tâches synthétiques style HashHop (Magic) ou BABILong étendu.

### Cible Phase 3

| Métrique | Seuil |
|---|---|
| BABILong @ 50M | ≥ 40% (cf. ARMT 80% sur GPT-2) |
| RULER @ 10M | ≥ 60 |
| Code-search @ 30M tokens (Linux kernel) | ≥ 50% precision@5 |
| KV memory @ 50M | ≤ 45 GB GPU + offload |

---

## 8. Évaluation — protocole rigoureux

**Pas de claim sans benchmark public.** Protocole :

1. **Suite obligatoire** :
   - RULER (NVIDIA) à 32K, 128K, 1M, 4M, 10M
   - MRCR-V2 (Gemini) 4-needle, 8-needle à 128K, 1M, 4M
   - BABILong à 1M, 5M, 10M, 50M
   - InfiniteBench long-code, long-math, long-retrieval
   - NoLiMa (no literal match) — exposer faiblesses lexicales

2. **Suite contrastive** :
   - Comparer à : Qwen2.5-7B-1M base (notre point de départ), Llama-4-Scout, MiniMax-M1, DeepSeek-V3.2 (sur 128K), un modèle dense 7B sans HELIX, et un baseline RAG sur le même 7B.

3. **Suite d'ablation** :
   - HELIX sans pyramide (juste hybrid backbone)
   - HELIX sans jetons-curseur
   - HELIX sans TTT
   - HELIX sans encoder JEPA (compression triviale)
   - Mesurer la contribution propre de chaque composant.

4. **Évaluation humaine** sur tâches pratiques : résumé de livres, questions sur monorepos, raisonnement multi-document long.

---

## 9. Risques et problèmes ouverts

| Risque | Mitigation |
|---|---|
| **Jetons-curseur ne convergent pas** sans supervision parfaite | Stratégie hybride : commencer par RL-from-AI-feedback (oracle GPT-4-class place les curseurs), passer en self-play après convergence. Backoff : si les curseurs n'aident pas, revenir à une attention purement implicite. |
| **Encoder/decoder JEPA introduit un goulot d'étranglement sémantique** | Tester l'objectif latent contre l'objectif token-level (LCC-style). Garder le meilleur. Risque réel — c'est la composante la plus inédite. |
| **TTT-E2E déstabilise la génération** (cas connu : "exploding" tests-time gradients) | Suivre exactement la recipe NVIDIA TTT-E2E (clip + LR scheduler dédié). Limiter mise à jour à des MLP isolés, jamais le tronc. |
| **Distillation hybride 75:25 perd trop de qualité @ 32K** | Test à mi-parcours, si régression > 3pt, augmenter ratio attention à 60:40 (cf. Samba 50:50). |
| **Données 5M+ n'existent pas à grande échelle** | Synthétiser : concatener livres, repo code, transcripts. Pas idéal, mais Nvidia UltraLong a montré que c'est suffisant à 4M. |
| **ARMT à 50M ne tient pas en 7B** (jamais testé à cette échelle) | Phase 3 = recherche, pas livraison. Falloir des expériences de scaling avant commitment full. |
| **Coût total ~75 000 H100-h** (~$3M @ $4/h) | Petit-lab budget. Demander 0.5–1M$ de cluster ou cloud sur 24 mois. |
| **Concurrence : DeepSeek V4, MiniMax-M2, etc. peuvent fermer le gap avant la livraison** | Le système est conçu pour être modulaire ; les composants (pyramide, RC-Head, TTT-MLP) sont valuables même si le backbone change. |

**Questions ouvertes (recherche fondamentale) :**
1. La compression JEPA récursive converge-t-elle vraiment sur des hiérarchies à 5+ niveaux ?
2. Les jetons-curseur peuvent-ils être appris **sans supervision explicite**, par RL sur la qualité de réponse seule ?
3. Le tronc dual-state SSM permet-il un "rewind" précis, ou seulement approximatif ?
4. La pyramide KV peut-elle être partagée **entre sessions** (mémoire à long terme cross-conversation) ?

---

## 10. Pourquoi maintenant — fenêtre stratégique

Trois facteurs convergent en mai 2026 pour rendre HELIX réalisable pour une équipe de 4–6 chercheurs :

1. **Tous les composants ont un prior art validé**. Hybrid backbone (Samba, Qwen3-Next, MiniMax), MLA (DeepSeek), pyramide (PyramidKV, R3Mem), TTT (NVIDIA TTT-E2E), épisodes (EM-LLM), distillation (MOHAWK, Mamba-in-the-Llama). HELIX est le premier à les **combiner avec les jetons-curseur** comme primitive d'attention.

2. **Les outils sont mûrs**. FlashAttention-3, vLLM, SGLang, kvpress, flash-linear-attention couvrent 80% de l'infra dont HELIX a besoin. Travail d'intégration, pas d'invention.

3. **Le marché bouge**. Anthropic vient de lancer une compaction API natale ; Magic.dev claim 100M ; DeepSeek V4 livre 1M flat-priced. La pression compétitive pour l'open-source > 1M est haute. **Le premier projet open-source avec >70% RULER @ 5M aura un impact disproportionné.**

---

## Références principales

- Qwen2.5-1M — https://arxiv.org/abs/2501.15383
- Mamba-in-the-Llama — https://arxiv.org/abs/2408.15237
- MOHAWK — https://arxiv.org/abs/2408.10189
- Llamba — https://arxiv.org/abs/2502.14458
- DeepSeek V3 (MLA) — https://arxiv.org/abs/2412.19437
- DeepSeek V3.2 (DSA) — https://arxiv.org/abs/2512.02556
- NSA — https://arxiv.org/abs/2502.11089
- MoBA — https://arxiv.org/abs/2502.13189
- Samba — https://arxiv.org/abs/2406.07522
- MiniMax-Text-01 — https://arxiv.org/abs/2501.08313
- Gated DeltaNet — https://arxiv.org/abs/2412.06464
- Qwen3-Next — https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct
- LongRoPE2 — https://arxiv.org/abs/2502.20082
- PoSE — https://arxiv.org/abs/2309.10400
- LongLoRA — https://arxiv.org/abs/2309.12307
- Nvidia UltraLong — https://arxiv.org/abs/2504.06214
- Llama 4 (iRoPE) — https://ai.meta.com/blog/llama-4-multimodal-intelligence/
- KVQuant — https://arxiv.org/abs/2401.18079
- KIVI — https://arxiv.org/abs/2402.02750
- PyramidKV — https://arxiv.org/abs/2406.02069
- R3Mem (compression réversible) — https://liner.com/review/r3mem-bridging-memory-retention-and-retrieval-via-reversible-compression
- Latent Context Compilation — https://arxiv.org/abs/2602.21221
- TTT-E2E (NVIDIA) — https://developer.nvidia.com/blog/reimagining-llm-memory-using-context-as-training-data-unlocks-models-that-learn-at-test-time/
- Titans — https://arxiv.org/abs/2501.00663 (composants à reprendre, mais ne pas s'y fier)
- EM-LLM — https://arxiv.org/abs/2407.09450
- ARMT — https://arxiv.org/abs/2407.04841
- Activation Beacon — https://arxiv.org/abs/2401.03462
- V-JEPA — https://www.emergentmind.com/topics/v-jepa-framework
- RULER — https://arxiv.org/abs/2404.06654
- BABILong — https://github.com/booydar/babilong
- SubQ (12M context, SSA) — https://explainx.ai/blog/subq-ssa-sparse-attention-12m-context-2026
