TRANSLATIONS = {
    "Español": {
        "title": "# 📚 Procesador de PDF con QA y Resumen",
        "upload_pdf": "Cargar PDF",
        "upload_images": "Cargar imágenes",
        "chunk_size": "Tamaño de chunk",
        "chunk_overlap": "Superposición de chunk",
        "process_btn": "Procesar",
        "processing_status": "Estado del procesamiento",
        "qa_tab": "Preguntas y Respuestas",
        "summary_tab": "Resumen",
        "chat_placeholder": "Haz una pregunta sobre el documento...",
        "chat_title": "Chat con el documento",
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
        "upload_pdf": "Upload PDF",
        "upload_images": "Upload Images",
        "chunk_size": "Chunk size",
        "chunk_overlap": "Chunk overlap",
        "process_btn": "Process",
        "processing_status": "Processing status",
        "qa_tab": "Questions and Answers",
        "summary_tab": "Summary",
        "chat_placeholder": "Ask a question about the document...",
        "chat_title": "Chat with document",
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
        "upload_pdf": "PDF hochladen",
        "upload_images": "Bilder hochladen",
        "chunk_size": "Chunk-Größe",
        "chunk_overlap": "Chunk-Überlappung",
        "process_btn": "PDF verarbe",
        "processing_status": "Verarbeitungsstatus",
        "qa_tab": "Fragen und Antworten",
        "summary_tab": "Zusammenfassung",
        "chat_placeholder": "Stellen Sie eine Frage zum Dokument...",
        "chat_title": "Chat mit Dokument",
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
        "upload_pdf": "Charger PDF",
        "upload_images": "Charger images",
        "chunk_size": "Taille du chunk",
        "chunk_overlap": "Chevauchement du chunk",
        "process_btn": "Traiter le",
        "processing_status": "État du traitement",
        "qa_tab": "Questions et Réponses",
        "summary_tab": "Résumé",
        "chat_placeholder": "Posez une question sur le document...",
        "chat_title": "Chat avec le document",
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
        "upload_pdf": "Carregar PDF",
        "upload_images": "Carregar imagens",
        "chunk_size": "Tamanho do chunk",
        "chunk_overlap": "Sobreposição do chunk",
        "process_btn": "Processar",
        "processing_status": "Status do processamento",
        "qa_tab": "Perguntas e Respostas",
        "summary_tab": "Resumo",
        "chat_placeholder": "Faça uma pergunta sobre o documento...",
        "chat_title": "Chat com o documento",
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




punctuation_dict = {
    "，": ",",
    "。": ".",
}
translation_table = str.maketrans(punctuation_dict)
stop_str = "<|im_end|>"

def render_ocr_text(text, result_path, format_text=False):
    if text.endswith(stop_str):
        text = text[: -len(stop_str)]
    text = text.strip()

    print(text)

    if format_text and "**kern" not in text:
        if "\\begin{tikzpicture}" not in text:
            html_path = "./render_tools/" + "/content-mmd-to-html.html"
            right_num = text.count("\\right")
            left_num = text.count("\left")

            if right_num != left_num:
                text = (
                    text.replace("\left(", "(")
                    .replace("\\right)", ")")
                    .replace("\left[", "[")
                    .replace("\\right]", "]")
                    .replace("\left{", "{")
                    .replace("\\right}", "}")
                    .replace("\left|", "|")
                    .replace("\\right|", "|")
                    .replace("\left.", ".")
                    .replace("\\right.", ".")
                )

            text = text.replace('"', "``").replace("$", "")

            outputs_list = text.split("\n")
            gt = ""
            for out in outputs_list:
                gt += '"' + out.replace("\\", "\\\\") + r"\n" + '"' + "+" + "\n"

            gt = gt[:-2]

            with open(html_path, "r") as web_f:
                lines = web_f.read()
                lines = lines.split("const text =")
                new_web = lines[0] + "const text =" + gt + lines[1]
        else:
            html_path = "./render_tools/" + "/tikz.html"
            text = text.translate(translation_table)
            outputs_list = text.split("\n")
            gt = ""
            for out in outputs_list:
                if out:
                    if (
                        "\\begin{tikzpicture}" not in out
                        and "\\end{tikzpicture}" not in out
                    ):
                        while out[-1] == " ":
                            out = out[:-1]
                            if out is None:
                                break

                        if out:
                            if out[-1] != ";":
                                gt += out[:-1] + ";\n"
                            else:
                                gt += out + "\n"
                    else:
                        gt += out + "\n"

            with open(html_path, "r") as web_f:
                lines = web_f.read()
                lines = lines.split("const text =")
                new_web = lines[0] + gt + lines[1]

        with open(result_path, "w") as web_f_new:
            web_f_new.write(new_web)
