# English/Hindi conversion summary

Key changes from the earlier prototype:

- replaced the previous non-target-language request defaults and examples
- supported application languages are now `en` and `hi`
- React UI has English/Hindi selection plus both voice and text query paths
- Groq Whisper receives `en` or `hi` using the selected target language
- Qwen receives an explicit requested answer language and English/Hindi/Hinglish grounding instructions
- no translation stage was added; BGE-M3 handles multilingual/cross-language retrieval
- Edge TTS now maps English to `en-IN-NeerjaNeural` and Hindi to `hi-IN-SwaraNeural`
- browser fallback uses `en-IN` or `hi-IN`
- feedback records the selected language for later fine-tuning
- included English, Hindi and bilingual sample knowledge files
- included a document extraction validator
- knowledge-folder `README.md` and hidden helper files are excluded from ingestion
- Phase-2 SFT examples are English/Hindi/Hinglish
- Phase-2 local ASR reference is Faster-Whisper
- Phase-2 local TTS reference is AI4Bharat Indic Parler-TTS
- added WER evaluation alongside CER
- added Windows PowerShell setup script
- added Groq key verification utility
