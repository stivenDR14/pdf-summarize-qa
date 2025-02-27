AI_MODELS = {
    "Huggingface / IBM granite granite 3.1 8b Instruct": "ibm-granite/granite-3.1-8b-instruct",
    "Huggingface / Mistral Small 24B Instruct": "mistralai/Mistral-Small-24B-Instruct-2501",
    "Huggingface / SmolLM2 1.7B Instruct": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "IBM Granite3.1 dense / Ollama local": "ollama",
    "Open AI / GPT-4o-mini": "openai",    
}

TRANSLATIONS = {
    "Español": {
        "title": "# 📚 Procesador de PDF con QA y Resumen",
        "api_key_required": "Para usar este modelo, necesitas una clave de API.",
        "model_type": "Tipo de modelo",
        "api_key_placeholder": "Ingresa tu clave de API",
        "project_id_placeholder": "Ingresa tu ID de proyecto",
        "ai_model": "Modelo AI",
        "upload_pdf": "Cargar PDF",
        "upload_images": "Cargar imágenes",
        "chunk_size": "Tamaño de chunk",
        "chunk_overlap": "Superposición de chunk",
        "process_btn": "Procesar",
        "processing_status": "Estado del procesamiento",
        "qa_tab": "Preguntas y Respuestas",
        "summary_tab": "Resumen",
        "chat_placeholder": "Haz una pregunta sobre el documento...",
        "chat_title": "Pregunta al documento",
        "generate_summary": "Generar Resumen",
        "summary_label": "Resumen del documento",
        "pdf_processed": "PDF procesado y almacenado correctamente",
        "load_pdf_first": "Por favor, carga un PDF primero.",
        "map_prompt": """Escribe un resumen conciso del siguiente texto:
        "{text}"
        RESUMEN CONCISO:""",
        "combine_prompt": """Escribe un resumen detallado basado en los siguientes resúmenes de diferentes secciones del texto:
        "{text}"
        RESUMEN DETALLADO:"""
    },
    "English": {
        "title": "# 📚 PDF Processor with QA and Summary",
        "api_key_required": "To use this model, you need an API key.",
        "model_type": "Model type",
        "api_key_placeholder": "Enter your API key",
        "project_id_placeholder": "Enter your project ID",
        "ai_model": "AI Model",
        "upload_pdf": "Upload PDF",
        "upload_images": "Upload Images",
        "chunk_size": "Chunk size",
        "chunk_overlap": "Chunk overlap",
        "process_btn": "Process",
        "processing_status": "Processing status",
        "qa_tab": "Questions and Answers",
        "summary_tab": "Summary",
        "chat_placeholder": "Ask a question about the document...",
        "chat_title": "Question to document",
        "generate_summary": "Generate Summary",
        "summary_label": "Document summary",
        "pdf_processed": "PDF processed and stored successfully",
        "load_pdf_first": "Please load a PDF first.",
        "map_prompt": """Write a concise summary of the following text:
        "{text}"
        CONCISE SUMMARY:""",
        "combine_prompt": """Write a detailed summary based on the following summaries from different sections of the text:
        "{text}"
        DETAILED SUMMARY:"""
    },
    "Deutsch": {
        "title": "# 📚 PDF-Prozessor mit Q&A und Zusammenfassung",
        "model_type": "Modelltyp",
        "api_key_required": "Um dieses Modell zu verwenden, benötigen Sie einen API-Schlüssel.",
        "api_key_placeholder": "API-Schlüssel eingeben",
        "project_id_placeholder": "Projekt-ID eingeben",
        "ai_model": "AI-Modell",
        "upload_pdf": "PDF hochladen",
        "upload_images": "Bilder hochladen",
        "chunk_size": "Chunk-Größe",
        "chunk_overlap": "Chunk-Überlappung",
        "process_btn": "PDF verarbe",
        "processing_status": "Verarbeitungsstatus",
        "qa_tab": "Fragen und Antworten",
        "summary_tab": "Zusammenfassung",
        "chat_placeholder": "Stellen Sie eine Frage zum Dokument...",
        "chat_title": "Frage zum Dokument",
        "generate_summary": "Zusammenfassung generieren",
        "summary_label": "Dokumentzusammenfassung",
        "pdf_processed": "PDF erfolgreich verarbeitet und gespeichert",
        "load_pdf_first": "Bitte laden Sie zuerst ein PDF hoch.",
        "map_prompt": """Schreiben Sie eine kurze Zusammenfassung des folgenden Textes:
        "{text}"
        KURZE ZUSAMMENFASSUNG:""",
        "combine_prompt": """Schreiben Sie eine detaillierte Zusammenfassung basierend auf den folgenden Zusammenfassungen verschiedener Textabschnitte:
        "{text}"
        DETAILLIERTE ZUSAMMENFASSUNG:"""
    },
    "Français": {
        "title": "# 📚 Processeur PDF avec QR et Résumé", 
        "model_type": "Type de modèle",
        "api_key_required": "Pour utiliser ce modèle, vous avez besoin d'une clé API.",
        "api_key_placeholder": "Entrez votre clé API",
        "project_id_placeholder": "Entrez votre ID de projet",
        "ai_model": "Modèle AI",
        "upload_pdf": "Charger PDF",
        "upload_images": "Charger images",
        "chunk_size": "Taille du chunk",
        "chunk_overlap": "Chevauchement du chunk",
        "process_btn": "Traiter le",
        "processing_status": "État du traitement",
        "qa_tab": "Questions et Réponses",
        "summary_tab": "Résumé",
        "chat_placeholder": "Posez une question sur le document...",
        "chat_title": "Question au document",
        "generate_summary": "Générer le résumé",
        "summary_label": "Résumé du document",
        "pdf_processed": "PDF traité et enregistré avec succès",
        "load_pdf_first": "Veuillez d'abord charger un PDF.",
        "map_prompt": """Écrivez un résumé concis du texte suivant :
        "{text}"
        RÉSUMÉ CONCIS :""",
        "combine_prompt": """Écrivez un résumé détaillé basé sur les résumés suivants de différentes sections du texte :
        "{text}"
        RÉSUMÉ DÉTAILLÉ :"""
    },
    "Português": {
        "title": "# 📚 Processador de PDF com P&R e Resumo",
        "model_type": "Tipo de modelo",
        "api_key_required": "Para usar este modelo, necesitas una clave de API.",
        "api_key_placeholder": "Digite sua chave API",
        "project_id_placeholder": "Digite seu ID de projeto",
        "ai_model": "Modelo AI",
        "upload_pdf": "Carregar PDF",
        "upload_images": "Carregar imagens",
        "chunk_size": "Tamanho do chunk",
        "chunk_overlap": "Sobreposição do chunk",
        "process_btn": "Processar",
        "processing_status": "Status do processamento",
        "qa_tab": "Perguntas e Respostas",
        "summary_tab": "Resumo",
        "chat_placeholder": "Faça uma pergunta sobre o documento...",
        "chat_title": "Pergunta ao documento",
        "generate_summary": "Gerar Resumo",
        "summary_label": "Resumo do documento",
        "pdf_processed": "PDF processado e armazenado com sucesso",
        "load_pdf_first": "Por favor, carregue um PDF primeiro.",
        "map_prompt": """Escreva um resumo conciso do seguinte texto:
        "{text}"
        RESUMO CONCISO:""",
        "combine_prompt": """Escreva um resumo detalhado baseado nos seguintes resumos de diferentes seções do texto:
        "{text}"
        RESUMO DETALHADO:"""
    }
} 