# Documentation d'intégration des agents AIPROD_V33

## Objectif

Orchestrer la génération d'une vidéo complète à partir d'un prompt utilisateur, en combinant tous les agents du pipeline.

## Architecture du pipeline

1. **Script Generator** : Génère le découpage en scènes et le texte de narration à partir du prompt.
2. **AudioGenerator** : Génère la narration (TTS) et les voix synchronisées.
3. **MusicComposer** : Génère la musique adaptée à chaque scène.
4. **VoiceDirector** : Gère l'émotion, le style et la synchronisation des voix.
5. **Image/Video Generator** : Génère les images ou clips vidéo pour chaque scène.
6. **Montage** : Concatène les clips vidéo, synchronise et mixe les pistes audio (narration, musique, bruitages).
7. **PostProcessor** : Applique transitions, effets, titrage, sous-titres, overlays 3D, etc.

## Exemple d'orchestration (pseudo-code)

```python
# 1. Génération du script/scènes
scenes = script_generator.generate(prompt)

# 2. Génération des voix
narration_audio = audio_generator.generate(scenes)

# 3. Génération de la musique
music_audio = music_composer.compose(scenes)

# 4. Génération des images/vidéos
video_clips = image_video_generator.generate(scenes)

# 5. Montage audio/vidéo
final_video = montage_agent.assemble(video_clips, narration_audio, music_audio, sound_effects)

# 6. Post-processing
manifest = {
    "video_path": final_video,
    "transitions": [...],
    "titles": [...],
    "subtitles": [...],
    "effects": [...],
    "overlays": [...]
}
post_processor = PostProcessor()
result = post_processor.run(manifest)
```

## Points clés

- Chaque agent doit recevoir les bons paramètres (scènes, timings, styles, etc.).
- Les fichiers intermédiaires (audio, vidéo) doivent être accessibles pour le montage et le post-processing.
- Le manifest du PostProcessor permet de piloter toutes les finitions vidéo.
- Le pipeline est modulaire : chaque agent peut être remplacé ou amélioré indépendamment.

## Conseils

- Utilisez des mocks pour tester chaque étape séparément.
- Vérifiez la synchronisation audio/vidéo avant le post-processing.
- Adaptez le manifest selon les besoins du projet (effets, overlays, titrage, etc.).

## Pour aller plus loin

- Ajoutez des agents pour la génération d'effets spéciaux, d'animations, ou d'intégration cloud.
- Intégrez une interface utilisateur pour piloter le pipeline à partir d'un prompt.
- Automatisez les tests et la validation de la vidéo finale.

---

Ce document peut être enrichi selon l'évolution du projet et des agents.
