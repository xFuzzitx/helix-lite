# Auto-critique de HELIX

*Mai 2026. Critique sans concession de la proposition `HELIX_architecture.md`. Objectif : identifier ce qui ne tient pas avant de construire, pas après.*

---

## TL;DR de la critique

HELIX combine 6 composants dont **3 sont solides** (backbone hybride, pyramide KV, distillation), **2 sont risqués** (TTT-E2E, encoder JEPA récursif), et **1 est probablement faux ou mal conçu** (jetons-curseur tels que présentés). Le **vrai goulot d'étranglement** que j'ai sous-estimé n'est ni l'architecture ni le budget compute, mais **la disponibilité de données 5M+ utiles**. Le système a aussi un problème de **scope explosif** — ~5 projets de recherche superposés, là où une équipe de 4–6 chercheurs en mène 1.

---

## 1. Critiques fondamentales (les piliers qui vacillent)

### 1.1 Les jetons-curseur sont mal posés

**Le problème de fond** : un jeton-curseur est une **action discrète** dans un modèle qui n'apprend bien que sur des objectifs **continus, différentiables, denses en signal**. Trois sous-problèmes :

**(a) Crédit-assignment** : le gain d'un `⟨ZOOM-IN⟩` ne se manifeste qu'à la **réponse finale**, parfois 1000 tokens plus tard. C'est exactement le problème que les modèles de raisonnement (o1, o3, Claude thinking) peinent à résoudre malgré des milliers de H100-h de RL. Prétendre le résoudre en mois 3 d'un projet 7B est naïf.

**(b) Espace d'action astronomique** : à chaque step, le modèle choisit (token normal | curseur × type × span_id). Si span_id désigne un chunk parmi 24 414 (à 50M / 2048), l'espace d'action est ~75 000. Aucun travail public n'a entraîné un transformer à choisir parmi 75 000 actions discrètes par token sans un signal de récompense bien plus dense que ce que je propose.

**(c) Le "teacher oracle" est circulaire** : ma stratégie de mitigation était "un GPT-4-class teacher place les curseurs optimaux". Mais si le teacher peut décider où regarder, **il a déjà résolu le problème de long-context**. Le student apprend à imiter une heuristique du teacher, pas à raisonner. Ce n'est pas généralisable.

**Verdict** : la primitive est intéressante mais la formulation actuelle ne marchera pas. **Reformulation possible** : au lieu d'un jeton émis dans la séquence de sortie, un **gating module** qui produit en parallèle un score continu de "zoom" pour chaque chunk, intégré différentiablement. C'est moins ambitieux mais entraînable. Effectivement, on retombe sur quelque chose proche de DSA top-k de DeepSeek V3.2 — donc la "nouveauté" disparaît.

### 1.2 L'encoder JEPA récursif suppose un tronc stable, mais le tronc est lui-même en cours d'entraînement

**Le problème** : pour entraîner l'encoder JEPA (objectif latent : prédire les embeddings du tronc), il faut des cibles `h_target`. Si le tronc est **gelé**, l'encoder est calibré sur une version figée. Si le tronc continue à apprendre (curriculum de longueur, fine-tuning curseurs, TTT-E2E), **les cibles se déplacent**. On entre en cycle interminable :

```
Train tronc → Train encoder → Tronc dérive → Re-train encoder → …
```

**Mitigation envisagée** : EMA target encoder (style BYOL, V-JEPA). Mais V-JEPA fonctionne parce que le tronc visuel converge sur des embeddings stables ; pour un LLM en continued-pretraining sur du long-context, la stabilité n'est pas démontrée.

**Vrai risque** : la compression hiérarchique récursive (N5→N4→N3→N2…) suppose que **chaque niveau** est stable assez longtemps pour entraîner le niveau au-dessus. À 5+ niveaux, c'est un château de cartes.

**Solution honnête** : limiter à **2 niveaux JEPA maximum** (chunk-level et episode-level), abandonner le N4/N5 méta-épisodes. Le gain de N5 sur 50M était purement spéculatif.

### 1.3 État dual rapide/lent : pas démontré que c'est exploitable

J'ai postulé un `h_slow` checkpointé aux frontières d'épisodes pour permettre un "rewind" O(1). En pratique :

- Les états SSM sont **continûment mis à jour** par les tokens. Restaurer `h_slow` d'un épisode passé puis re-stream les tokens depuis ce point coûte O(taille_épisode), pas O(1).
- Si on **n'effectue pas le re-stream** et qu'on lit directement `h_slow`, l'état ne reflète pas le contexte intermédiaire — donc le modèle "saute" du présent à un état passé sans transition cohérente.
- Aucun travail public ne montre qu'un Mamba ou GDN peut faire des sauts d'état arbitraires sans dégrader la génération.

**Verdict** : le mécanisme tel que décrit est probablement faux. **Reformulation** : utiliser `h_slow` comme **input additionnel** aux couches d'attention conservées (memory-as-context style Titans), sans prétendre faire un rewind.

### 1.4 Le ratio 75:25 GDN:attention peut être mauvais avec retrieval-driven design

Le ratio 75:25 vient de Qwen3-Next, **mais Qwen3-Next n'a pas de récupération explicite**. L'attention y sert d'ancrage de recall implicite. Avec HELIX qui ajoute curseurs + pyramide + TTT, on a **trois mécanismes redondants** pour le recall :
1. L'attention sparse (8/32 couches)
2. Les curseurs émis par le RC-Head
3. La TTT-MLP qui apprend en cours de route

C'est probablement **trop de redondance**, pas assez d'attention. Si je devais parier : pour le recall multi-hop à 5M+, il faudrait plutôt **50:50 ou 60:40** vers l'attention. Mais alors le coût attention explose et l'argument sous-quadratique s'affaiblit.

**Aucun choix de ratio n'est défendu empiriquement dans le doc** — c'est un copier-coller de Qwen3-Next sans valider la transposabilité.

---

## 2. Critiques techniques (math et infra)

### 2.1 La math de mémoire est fausse

J'ai écrit pour N3 (50M tokens, latents JEPA) :

> "50M / chunk_size(2048) × 64 latents × 32 × 128 fp16 ≈ 50M × ~2 KB = 100 MB par layer × 32 = 3.2 GB"

Recalcul honnête :
- 50M / 2048 = 24 414 chunks
- 24 414 × 64 latents = **1.56M latents par layer**
- Chaque latent : 128 dim × 2 bytes = 256 bytes
- Par layer : 1.56M × 256 = **400 MB**
- 32 layers : **12.8 GB**

J'avais sous-estimé d'un facteur 4×. Total réel : ~36 GB GPU au lieu de ~27 GB. **Tient encore sur H100 80GB** mais avec moins de marge pour les activations.

**Pire** : si les latents JEPA sont produits par un encoder qui a sa propre dimension (256 ou 512), pas 128, le coût double encore. Je n'ai pas spécifié.

**Pire encore** : N3 doit-il être par-couche ou partagé ? Si partagé, 400 MB total — bonus. Si par-couche (parce que chaque couche a des features différentes), 12.8 GB. **Le doc ne tranche pas.** C'est une ambiguïté centrale qui change le coût mémoire d'un ordre de grandeur.

### 2.2 Le budget compute est sous-estimé d'un facteur 2–3×

J'ai chiffré 75 000 H100-h sur 24 mois. Postulats :
- Distillation : 10 000 H100-h pour ~10B tokens. **Réaliste**.
- Pyramide JEPA : 3 000 H100-h. **Réaliste**.
- Curseurs : 7 000 H100-h. **Probablement 5–10× sous-estimé** si on doit faire du RL pour leur émission.
- Curriculum 32K → 5M : 5 000 H100-h. **Sous-estimé** : Nvidia UltraLong = 3 300 H100-h pour 4M sur un seul palier final ; HELIX a 4 paliers.
- Phase 2 et 3 : "+50K" cumulé. Très flou.

**Estimation honnête** : 200 000 H100-h sur 24 mois (~$8M @ $4/h cloud, ~$3M sur infra propre). C'est une équipe de **labo bien financé**, pas une équipe de 4–6 chercheurs.

### 2.3 L'engineering custom-kernel est massivement sous-estimé

J'ai mentionné "vLLM-fork ou SGLang-fork" comme une option. Réalité :
- Pyramide promotion/dégradation = nouveau scheduler de KV avec invariants complexes
- JEPA decoder appelé en cours d'inférence = nouveau path d'execution
- Cursor token routing = nouveau mask construction par token
- TTT-E2E gradient updates pendant inference = nouveau memory management

**Conservateur : 30% du budget total est de l'infra custom**, pas de la recherche. Or je n'ai alloué aucun budget infra explicitement. C'est ~25 000 H100-h fantômes ou un ingé senior fullstack pour 18 mois (~$500K).

### 2.4 La distillation hybride régresse sur long context

J'ai cité Mamba-in-the-Llama qui retient ¼ d'attention. Mais :
- L'éval de Mamba-in-the-Llama monte jusqu'à **20× la longueur de distillation**, soit ~80K à partir de 4K.
- Aucune publication ne montre la conservation de qualité **à 1M+** après une distillation hybride.
- Toutes les conversions hybrides (Llamba, MOHAWK) **régressent** sur les benchmarks de raisonnement complexe (GSM8K, MMLU difficile) de 2–10pt.

**HELIX phase 1 part avec un modèle hybride dégradé** par rapport au Qwen2.5-7B-1M dense. Pour récupérer le score base, il faut **plus** de continued pretraining que ce que j'ai budgété.

---

## 3. Critiques stratégiques

### 3.1 Pourquoi pas RAG (mieux fait) ?

Mon doc balaie RAG en disant "fallback for static corpora". Mais en 2026 :
- RAG est ~1250× moins cher par requête (LaRA, ICML 2025)
- RAG a 30–45× moins de latence
- **Pour 80% des cas d'usage long-context réels** (codebase, email, agents), RAG match ou bat le long-context

**HELIX a un avantage étroit** : tâches où l'info est **dispersée sur tout le contexte** et nécessite synthèse cross-segment (pas retrieval). Exemples : résumé de livre, audit de codebase complet, analyse de toute une carrière email.

Mais pour ces tâches, **est-ce que 50M tokens est même la bonne unité ?** Un livre fait 100K. Un repo fait <5M. Une vie d'email fait peut-être 10M. **Le marché 50M+ est largement inventé**.

**Question difficile** : si on avait du RAG sophistiqué (Karpathy 3-layer pattern, LongRAG, hierarchical summarization) ET un modèle long-context à 1M, est-ce qu'on a vraiment besoin de HELIX-50M ? Probablement non pour la majorité des usages réels. **HELIX optimise une métrique au détriment de l'utilité.**

### 3.2 Le problème de données est insoluble par l'architecture

J'ai écrit "concatener livres, repos, transcripts" pour les données 5M+. Réalité :
- **Aucune source naturelle** ne fournit 5M+ tokens de **structure cohérente** (raisonnement long, références internes, voix consistante).
- **La concaténation** crée des "needle-in-haystack textures", pas du long-context organique.
- Les modèles entraînés sur des données concaténées **apprennent à trouver des aiguilles**, pas à raisonner sur 5M tokens.

C'est une limitation **fondamentale** que ni HELIX ni aucune autre architecture ne peut résoudre. RULER et NIAH récompensent les modèles qui sont bons sur des données synthétiques. **Les benchmarks long-context mesurent largement la capacité à exploiter des données concaténées artificiellement.**

Un modèle peut atteindre 80% RULER à 5M et être nul sur du long-context utile. C'est probablement le cas de Llama 4 Scout — claim 10M, eff. ~5M sur RULER, mais utilité réelle douteuse.

**Critique structurelle** : HELIX optimise pour exploiter de la donnée que la civilisation produit à peine. L'investissement aurait peut-être plus de ROI sur un meilleur RAG + un 1M context solide.

### 3.3 Scope explosif pour une équipe de 4–6 chercheurs

Composants HELIX, chacun un projet de recherche significatif :

| Composant | Effort réaliste | Risque |
|---|---|---|
| Distillation hybride 75:25 | 1 chercheur × 6 mois | Faible (precedent solide) |
| Pyramide KV multi-résolution | 1 chercheur × 9 mois | Moyen |
| Encoder/decoder JEPA récursif | 1 chercheur × 12 mois | Élevé |
| Jetons-curseur + RL emission | 2 chercheurs × 18 mois | Très élevé |
| TTT-E2E par épisode | 1 chercheur × 12 mois | Élevé |
| Custom inference engine | 1 ingé infra × 18 mois | Élevé |

**Total : ~7 ETP × 18 mois = 105 personne-mois**, là où j'ai implicitement supposé 4–6 chercheurs × 24 mois = 96–144 personne-mois — sans compter l'infra. Il manque ~30% d'effectif rien qu'à l'estimation.

**Plus dangereux** : ces composants ont des **interdépendances**. La pyramide dépend du JEPA encoder. Les curseurs dépendent du RC-Head qui dépend du tronc distillé. Si un composant pivote (ex. on abandonne JEPA latent pour token-level), tous les autres doivent re-converger. C'est le **anti-pattern "big bang integration"**.

### 3.4 Pas de falsifiabilité énoncée

Je n'ai pas dit **ce qui prouverait que HELIX est faux**. Bonne recherche = "on abandonne si X". Énoncés que j'aurais dû écrire :

- *Si après 5B tokens de distillation hybride, RULER@32K a régressé de >5pt vs Qwen2.5-7B-1M base, le ratio 75:25 est mauvais → revenir à 50:50.*
- *Si après 1B tokens de RL sur curseurs, le RC-Head n'émet pas les curseurs aux positions de l'oracle dans >70% des cas, la primitive ne convergera pas → abandonner et utiliser DSA top-k differentiable.*
- *Si l'encoder JEPA récursif ne reconstruit pas les latents au-delà de 2 niveaux avec précision >90%, abandonner les niveaux N4/N5.*

Sans ces gates explicites, le projet "sunken cost" peut continuer 18 mois avant qu'on admette l'échec.

---

## 4. Ce qui survit à la critique

Toute critique n'est pas nihilisme. Ce qui reste solide :

### 4.1 Backbone hybride 75:25 (avec ratio à valider)

- **Précédents production** : Qwen3-Next, MiniMax-Text-01, Samba.
- **Distillation** : MOHAWK et Mamba-in-the-Llama prouvent la transition.
- **Risque** : ratio peut être à ajuster, mais le cadre est solide.

### 4.2 Pyramide KV à 2–3 niveaux (pas 4)

- **Précédents** : PyramidKV, R3Mem, KVQuant nuq2.
- **Si on abandonne N3 latents JEPA et qu'on reste à N0 fp16 / N1 4-bit / N2 2-bit**, la pyramide est essentiellement KVQuant + Quest + LayerKV — déjà éprouvé.
- **Le niveau N3 latent est la partie spéculative**. Garder sans inflated claims.

### 4.3 MLA + iRoPE + curriculum progressif

- **Précédents** : DeepSeek V3, Llama 4 Scout, Qwen2.5-1M.
- **Cadre établi**. C'est de l'engineering, pas de la recherche.

### 4.4 La question structurante : "comment piloter la mémoire par le raisonnement"

C'est la **bonne question** posée par HELIX. Les jetons-curseur tels que formulés ne marchent probablement pas, mais la question reste **non résolue par le SOTA**. Une formulation correcte aurait de la valeur — peut-être :

- **Gating différentiable par chunk** (style mixture-of-experts attention)
- **Supervision par les attention scores eux-mêmes** (auto-bootstrap)
- **Distillation depuis un modèle de raisonnement à long thinking-budget** qui dit lui-même "je vais regarder la section X"

---

## 5. Ce que je ferais différemment

### Stratégie "shrinking circle"

Au lieu d'attaquer 5/10/50M en une architecture monolithique, **commencer petit et durcir** :

**Mois 1–6 : minimum viable HELIX**
- Backbone hybride 75:25 (distillation Qwen2.5-7B-1M).
- Pyramide à 2 niveaux : N0 fp16 (32K) + N1 KVQuant 2-bit (1M).
- Inference stack standard (vLLM + KVQuant + Quest).
- **Cible** : matcher Qwen2.5-7B-1M au benchmark, à throughput 2–3× supérieur.
- **Si ce n'est pas atteint en mois 6, le reste du projet est compromis.** Ne pas continuer.

**Mois 7–12 : la première vraie nouveauté**
- Choisir **un seul** des trois composants spéculatifs (curseurs, JEPA récursif, TTT-par-épisode).
- L'isoler dans un benchmark synthétique focalisé.
- Décider à mois 12 si ça marche.

**Mois 13–24 : intégration progressive**
- Si la nouveauté marche, l'intégrer dans le backbone.
- Si pas, choisir la suivante.
- Ne **jamais** intégrer 2 nouveautés non-validées en même temps.

### Le composant à essayer en premier

Si je dois parier : **TTT-E2E par épisode**. Justifications :
- Validé par NVIDIA (TTT-E2E plat) en 2026.
- L'extension "par épisode" est un delta naturel.
- Si ça ne marche pas, on apprend quelque chose de précis sur la mémoire test-time.
- N'invalide pas le reste du système si abandonné.

Les jetons-curseur sont **trop risqués pour être le premier delta**.

### Honnêteté sur le marché

Un modèle 7B qui fait **vraiment** 1M avec >85% RULER, 60% MRCR-V2, **et qui s'inférence sur un seul H100 à throughput compétitif** est déjà un product massif. Pas besoin de 50M. La phase 1 réussie de HELIX, sans phases 2/3, est plus précieuse pour la communauté que "HELIX-50M qui marche à 40%".

---

## 6. Récapitulatif honnête

| Composant HELIX | Verdict |
|---|---|
| Backbone hybride 75:25 | **Solide**, mais ratio à valider empiriquement |
| Pyramide KV multi-résolution (2–3 niveaux) | **Solide**, prior art validé |
| Pyramide à 4–5 niveaux avec JEPA récursif | **Spéculatif**, garder à 2 niveaux max |
| Jetons-curseur natifs émis dans le vocab | **Probablement faux** comme formulé. Reformuler en gating différentiable. |
| État dual rapide/lent SSM avec rewind | **Mécanisme mal posé**. Reformuler en memory-as-context. |
| TTT-E2E par épisode | **Risqué mais valuable**. Bon premier delta novateur à essayer. |
| Phase 3 — 50M | **Recherche, pas livraison.** Retirer des deliverables planifiés. |
| Budget 75K H100-h | **2–3× sous-estimé.** Vrai chiffre : 150–250K H100-h. |
| Scope global pour 4–6 chercheurs | **Trop ambitieux.** Réduire à un MVP + 1 nouveauté. |

**Le projet ne devrait pas être lancé tel quel.** Mais le **cadrage** (les bonnes questions, les bonnes références) est valable. Ce qu'il faut, c'est une **version 50–70% plus petite et 100% plus falsifiable**.
