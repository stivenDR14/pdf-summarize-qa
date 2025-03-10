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
        "chat_btn": "Preguntar",
        "generate_summary": "Generar Resumen",
        "summary_label": "Resumen del documento",
        "pdf_processed": "PDF procesado y almacenado correctamente",
        "load_pdf_first": "Por favor, carga un PDF primero.",
        "map_prompt": """Escribe un resumen conciso del siguiente texto:
        "{text}"
        RESUMEN CONCISO:""",
        "combine_prompt": """Escribe un resumen detallado basado en los siguientes resúmenes de diferentes secciones del texto:
        "{text}"
        RESUMEN DETALLADO:""",
        "mini_summary_title": "Resúmenes de cada fragmento",
        "mini_analysis_title": "Análisis de cada fragmento",
        "specialist_tab": "Asesor a tu medida",
        "specialist_title": "Asesor a tu medida",
        "specialist_label": "Establece el comportamiento y rol de tu asesor. Ej: Eres un especialista de finanzas que ayuda a interpretar los datos de un reporte financiero. A partir del documento y tu basta experiencia cuéntame que oportunidades y riesgos ves al invertir en lo que te proponen.",
        "specialist_output": "Respuesta de tu asesor",
        "specialist_btn": "Generar Respuesta"
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
        "chat_btn": "Ask",
        "generate_summary": "Generate Summary",
        "summary_label": "Document summary",
        "pdf_processed": "PDF processed and stored successfully",
        "load_pdf_first": "Please load a PDF first.",
        "map_prompt": """Write a concise summary of the following text:
        "{text}"
        CONCISE SUMMARY:""",
        "combine_prompt": """Write a detailed summary based on the following summaries from different sections of the text:
        "{text}"
        DETAILED SUMMARY:""",
        "mini_summary_title": "Summaries of each fragment",
        "mini_analysis_title": "Analysis of each fragment",
        "specialist_tab": "Customized Advisor",
        "specialist_title": "Customized Advisor",
        "specialist_label": "Set the behavior and role of your advisor. Example: You are a financial expert who helps interpret the data of a financial report. Based on the document and your extensive experience, tell me what opportunities and risks you see in what they propose.",
        "specialist_output": "Answer of your advisor",
        "specialist_btn": "Generate Answer"
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
        "chat_btn": "Fragen",
        "generate_summary": "Zusammenfassung generieren",
        "summary_label": "Dokumentzusammenfassung",
        "pdf_processed": "PDF erfolgreich verarbeitet und gespeichert",
        "load_pdf_first": "Bitte laden Sie zuerst ein PDF hoch.",
        "map_prompt": """Schreiben Sie eine kurze Zusammenfassung des folgenden Textes:
        "{text}"
        KURZE ZUSAMMENFASSUNG:""",
        "combine_prompt": """Schreiben Sie eine detaillierte Zusammenfassung basierend auf den folgenden Zusammenfassungen verschiedener Textabschnitte:
        "{text}"
        DETAILLIERTE ZUSAMMENFASSUNG:""",
        "mini_summary_title": "Zusammenfassungen von jedem Fragment",
        "mini_analysis_title": "Analyse von jedem Fragment",
        "specialist_tab": "Anpassbarer Berater",
        "specialist_title": "Anpassbarer Berater",
        "specialist_label": "Setzen Sie das Verhalten und die Rolle Ihres Beraters fest. Beispiel: Sie sind ein Finanzexperte, der bei der Interpretation von Finanzdaten aus einem Bericht hilft. Basierend auf dem Dokument und Ihrer umfassenden Erfahrung, erzählen Sie mir, was Sie in dem sehen, was sie Ihnen vorschlagen.",
        "specialist_output": "Antwort Ihres Beraters",
        "specialist_btn": "Antwort generieren"
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
        "chat_btn": "Poser une question",
        "generate_summary": "Générer le résumé",
        "summary_label": "Résumé du document",
        "pdf_processed": "PDF traité et enregistré avec succès",
        "load_pdf_first": "Veuillez d'abord charger un PDF.",
        "map_prompt": """Écrivez un résumé concis du texte suivant :
        "{text}"
        RÉSUMÉ CONCIS :""",
        "combine_prompt": """Écrivez un résumé détaillé basé sur les résumés suivants de différentes sections du texte :
        "{text}"
        RÉSUMÉ DÉTAILLÉ :""",
        "mini_summary_title": "Résumés de chaque fragment",
        "mini_analysis_title": "Analyse de chaque fragment",
        "specialist_tab": "Conseiller personnalisé",
        "specialist_title": "Conseiller personnalisé",
        "specialist_label": "Définissez le comportement et le rôle de votre conseiller. Exemple : Vous êtes un expert financier qui aide à interpréter les données d'un rapport financier. Basé sur le document et votre vaste expérience, partagez-moi ce que vous voyez dans ce qu'ils vous proposent.",
        "specialist_output": "Réponse de votre conseiller",
        "specialist_btn": "Générer la réponse"
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
        "chat_btn": "Perguntar",
        "generate_summary": "Gerar Resumo",
        "summary_label": "Resumo do documento",
        "pdf_processed": "PDF processado e armazenado com sucesso",
        "load_pdf_first": "Por favor, carregue um PDF primeiro.",
        "map_prompt": """Escreva um resumo conciso do seguinte texto:
        "{text}"
        RESUMO CONCISO:""",
        "combine_prompt": """Escreva um resumo detalhado baseado nos seguintes resumos de diferentes seções do texto:
        "{text}"
        RESUMO DETALHADO:""",
        "mini_summary_title": "Resúmenes de cada fragmento",
        "mini_analysis_title": "Análisis de cada fragmento",
        "specialist_tab": "Assistente Personalizado",
        "specialist_title": "Assistente Personalizado",
        "specialist_label": "Defina o comportamento e o papel do seu assistente. Exemplo: Você é um especialista em finanças que ajuda a interpretar os dados de um relatório financeiro. Com base no documento e em sua ampla experiência, compartilhe comigo o que você vê naquilo que eles lhe propõem.",
        "specialist_output": "Resposta do seu assistente",
        "specialist_btn": "Gerar Resposta"
    }
} 