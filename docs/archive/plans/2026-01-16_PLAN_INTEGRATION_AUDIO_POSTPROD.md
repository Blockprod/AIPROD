# Plan d’intégration des modules audio et post-production – AIPROD V33

## Objectif

Compléter la pipeline AIPROD V33 pour couvrir l’ensemble des besoins d’un studio vidéo automatisé, en ajoutant :

- AudioGenerator (TTS + musique)
- PostProcessor (montage, transitions, effets, titrage)
- VoiceDirector (direction vocale, émotions, synchronisation)
- MusicComposer (bande-son contextuelle)

---

## 1. AudioGenerator

- **Fonction** : Générer la voix off (TTS) et la musique de fond.
- **Actions** :
  - Intégrer ElevenLabs et Google TTS via API (wrapper Python).
  - Ajouter un agent `AudioGenerator` dans `src/agents/audio_generator.py`.
  - Exposer les paramètres (langue, style, émotion) dans le manifest créatif.
  - Générer la musique via API (Soundful, AIVA, ou Google MusicLM si dispo).
  - Stocker les assets audio dans le manifest et les injecter dans le pipeline.

## 2. PostProcessor

- **Fonction** : Appliquer montage, transitions, effets, titrage, sous-titres.
- **Actions** :
  - Créer un agent `PostProcessor` dans `src/agents/post_processor.py`.
  - Utiliser MoviePy, FFmpeg ou APIs cloud (Runway, Veo) pour le montage.
  - Ajouter la gestion des transitions, effets visuels, titrage (overlay texte).
  - Générer automatiquement les sous-titres à partir du script (STT ou TTS).
  - Intégrer ce module à la fin du pipeline, juste avant la QA technique.

## 3. VoiceDirector

- **Fonction** : Contrôler l’intonation, l’émotion, la synchronisation audio/vidéo.
- **Actions** :
  - Créer un agent `VoiceDirector` dans `src/agents/voice_director.py`.
  - Mapper les émotions/scènes à des paramètres TTS (prosodie, pauses, intensité).
  - Synchroniser la voix avec les scènes (timestamps, cues).
  - Ajouter la gestion des retakes/ajustements via ICC si besoin.

## 4. MusicComposer

- **Fonction** : Générer une bande-son adaptée au contenu/scènes.
- **Actions** :
  - Créer un agent `MusicComposer` dans `src/agents/music_composer.py`.
  - Intégrer une API de génération musicale (AIVA, Soundful, MusicLM, etc.).
  - Adapter la musique au rythme, ambiance, transitions du script.
  - Injecter la musique dans le manifest et la timeline vidéo.

---

## Intégration dans le pipeline

- Mettre à jour le state machine (`src/orchestrator/state_machine.py`) pour inclure les nouveaux états :
  - `AUDIO_GENERATION`, `VOICE_DIRECTION`, `MUSIC_COMPOSITION`, `POST_PROCESSING`
- Ajouter les transitions dans `src/orchestrator/transitions.py`.
- Étendre le manifest créatif pour inclure les assets audio, musique, sous-titres.
- Ajouter des tests unitaires et d’intégration pour chaque module.
- Mettre à jour la documentation technique et API.

---

## Roadmap indicative

1. AudioGenerator (TTS + musique) – 3 à 5 jours
2. MusicComposer – 2 à 3 jours
3. VoiceDirector – 2 jours
4. PostProcessor – 4 à 6 jours
5. Intégration pipeline, tests, docs – 3 jours

---

**Résultat attendu : pipeline vidéo IA 100% automatisée, de la génération à la post-production, avec contrôle créatif complet.**
