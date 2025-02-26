import gradio as gr
from pdf_processor import PDFProcessor
from utils import TRANSLATIONS

class PDFProcessorUI:
    def __init__(self):
        self.processor = PDFProcessor()
        self.current_language = "English"
    
    def change_language(self, language):
        self.current_language = language
        self.processor.set_language(language)
        
        # Retornamos todos los textos que necesitan ser actualizados
        return [
            TRANSLATIONS[language]["title"],
            gr.update(label=TRANSLATIONS[language]["upload_pdf"]),
            gr.update(label=TRANSLATIONS[language]["upload_images"]),
            gr.update(label=TRANSLATIONS[language]["chunk_size"]),
            gr.update(label=TRANSLATIONS[language]["chunk_overlap"]),
            gr.update(value=TRANSLATIONS[language]["process_btn"]),
            gr.update(label=TRANSLATIONS[language]["processing_status"]),
            gr.update(label=TRANSLATIONS[language]["qa_tab"]),
            gr.update(label=TRANSLATIONS[language]["summary_tab"]),
            gr.update(placeholder=TRANSLATIONS[language]["chat_placeholder"]),
            TRANSLATIONS[language]["chat_title"],
            gr.update(value=TRANSLATIONS[language]["generate_summary"]),
            gr.update(label=TRANSLATIONS[language]["summary_label"])
        ]
    
    def process_pdf(self, pdf_file, image_file, chunk_size, chunk_overlap):
        return self.processor.process_pdf(pdf_file, image_file, chunk_size, chunk_overlap)
    
    def qa_interface(self, message, history):
        return self.processor.get_qa_response(message, history)
    
    def summarize_interface(self):
        return self.processor.get_summary()
    

    def upload_file(files):
        file_paths = [file.name for file in files]
        return file_paths[0]
    
    def create_ui(self):
        with gr.Blocks() as demo:
            title = gr.Markdown(TRANSLATIONS[self.current_language]["title"])
            
            with gr.Row():
                language_dropdown = gr.Dropdown(
                    choices=list(TRANSLATIONS.keys()),
                    value=self.current_language,
                    label="Language/Idioma/Sprache/Langue/Língua"
                )
            
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        pdf_file = gr.File(
                            label=TRANSLATIONS[self.current_language]["upload_pdf"],
                            file_types=[".pdf"]
                        )
                        with gr.Column():
                            image_file = gr.File(
                                label=TRANSLATIONS[self.current_language]["upload_images"],
                                file_types=[".png"]
                            )
                            """ upload_button = gr.UploadButton(TRANSLATIONS[self.current_language]["upload_images"], file_types=["image", "video"], file_count="single")
                            upload_button.upload(self.upload_file, inputs=[upload_button], outputs=[image_file]) """
                    chunk_size = gr.Number(
                        value=1000,
                        label=TRANSLATIONS[self.current_language]["chunk_size"]
                    )
                    chunk_overlap = gr.Number(
                        value=100,
                        label=TRANSLATIONS[self.current_language]["chunk_overlap"]
                    )
                    process_btn = gr.Button(
                        TRANSLATIONS[self.current_language]["process_btn"]
                    )
                    process_output = gr.Textbox(
                        label=TRANSLATIONS[self.current_language]["processing_status"]
                    )
            
            with gr.Tabs() as tabs:
                qa_tab = gr.Tab(TRANSLATIONS[self.current_language]["qa_tab"])
                summary_tab = gr.Tab(TRANSLATIONS[self.current_language]["summary_tab"])
            
            with qa_tab:
                chat_title = gr.Markdown(TRANSLATIONS[self.current_language]["chat_title"])
                chat_placeholder = gr.Textbox(
                    placeholder=TRANSLATIONS[self.current_language]["chat_placeholder"],
                    container=False,
                    show_label=False
                )
                chatbot = gr.Markdown(height=400)
            
            with summary_tab:
                summarize_btn = gr.Button(
                    TRANSLATIONS[self.current_language]["generate_summary"]
                )
                summary_output = gr.Textbox(
                    label=TRANSLATIONS[self.current_language]["summary_label"],
                    lines=10
                )
            
            # Eventos
            language_dropdown.change(
                fn=self.change_language,
                inputs=[language_dropdown],
                outputs=[
                    title,
                    pdf_file,
                    image_file,
                    chunk_size,
                    chunk_overlap,
                    process_btn,
                    process_output,
                    qa_tab,
                    summary_tab,
                    chat_placeholder,
                    chat_title,
                    summarize_btn,
                    summary_output
                ]
            )
            
            # Evento de chat
            chat_placeholder.submit(
                fn=self.qa_interface,
                inputs=[chat_placeholder, chatbot],
                outputs=[chatbot]
            )
            
            # Otros eventos
            process_btn.click(
                fn=self.process_pdf,
                inputs=[pdf_file, image_file, chunk_size, chunk_overlap],
                outputs=[process_output]
            )
            
            summarize_btn.click(
                fn=self.summarize_interface,
                inputs=[],
                outputs=[summary_output]
            )
        
        return demo

if __name__ == "__main__":
    ui = PDFProcessorUI()
    demo = ui.create_ui()
    demo.launch()
