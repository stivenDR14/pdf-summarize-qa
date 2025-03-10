import gradio as gr
from pdf_processor import PDFProcessor
from utils import AI_MODELS, TRANSLATIONS

class PDFProcessorUI:
    def __init__(self):
        self.processor = PDFProcessor()
        self.current_language = "English"
        self.current_ai_model = "Huggingface / IBM granite granite 3.1 8b Instruct"
        self.current_type_model = "Api Key"
    
    def change_language(self, language):
        self.current_language = language
        self.processor.set_language(language)
        
        # Retornamos todos los textos que necesitan ser actualizados
        return [
            TRANSLATIONS[language]["title"],
            gr.update(label=TRANSLATIONS[language]["upload_pdf"]),
            gr.update(label=TRANSLATIONS[language]["chunk_size"]),
            gr.update(label=TRANSLATIONS[language]["chunk_overlap"]),
            gr.update(value=TRANSLATIONS[language]["process_btn"]),
            gr.update(label=TRANSLATIONS[language]["processing_status"]),
            gr.update(label=TRANSLATIONS[language]["qa_tab"]),
            gr.update(label=TRANSLATIONS[language]["summary_tab"]),
            gr.update(label=TRANSLATIONS[language]["specialist_tab"]),
            gr.update(label=TRANSLATIONS[language]["mini_summary_title"]),
            gr.update(label=TRANSLATIONS[language]["mini_analysis_title"]),
            gr.update(placeholder=TRANSLATIONS[language]["chat_placeholder"]),
            TRANSLATIONS[language]["chat_title"],
            gr.update(value=TRANSLATIONS[language]["chat_btn"]),
            gr.update(value=TRANSLATIONS[language]["generate_summary"]),
            gr.update(label=TRANSLATIONS[language]["summary_label"]),
            gr.update(label=TRANSLATIONS[language]["ai_model"]),
            TRANSLATIONS[language]["specialist_title"],
            gr.update(label=TRANSLATIONS[language]["specialist_label"]),
            gr.update(label=TRANSLATIONS[language]["specialist_output"]),
            gr.update(value=TRANSLATIONS[language]["specialist_btn"])
        ]
    
    def change_ai_model(self, ai_model):
        self.current_ai_model = ai_model
        if ai_model == "IBM Granite3.1 dense / Ollama local":
            return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False, maximum=2048), gr.update(visible=False, maximum=200)
        elif ai_model == "Open AI / GPT-4o-mini":
            return gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), gr.update(visible=False, maximum=2048), gr.update(visible=False, maximum=200)
        else:
            return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False, maximum=500), gr.update(visible=False, maximum=100)
    
    def change_type_model(self, type_model):
        self.current_type_model = type_model
        if type_model == "Api Key":
            if self.current_ai_model == "IBM Granite3.1 dense / Ollama local":
                return gr.update(visible=False), gr.update(visible=False)
            else:
                return gr.update(visible=True), gr.update(visible=False)
        else:
            return gr.update(visible=False), gr.update(visible=False)
    
    def process_pdf(self, pdf_file, chunk_size, chunk_overlap, ai_model, type_model, api_key, project_id_watsonx):
        return self.processor.process_pdf(pdf_file, chunk_size, chunk_overlap, ai_model, type_model, api_key, project_id_watsonx)
    
    def qa_interface(self, message, history, ai_model, type_model, api_key, project_id_watsonx):
        return self.processor.get_qa_response(message, history, ai_model, type_model, api_key, project_id_watsonx)
    
    def summarize_interface(self, ai_model, type_model, api_key, project_id_watsonx):
        return self.processor.get_summary(ai_model, type_model, api_key, project_id_watsonx)
    
    def specialist_opinion(self, ai_model, type_model, api_key, project_id_watsonx, specialist_prompt):
        return self.processor.get_specialist_opinion(ai_model, type_model, api_key, project_id_watsonx, specialist_prompt)
    
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
                    label="Language/Idioma/Sprache/Langue/Língua",
                    key="language_dropdown" 
                )
                ai_model_dropdown = gr.Dropdown(
                    choices=list(AI_MODELS.keys()),
                    value=self.current_ai_model,
                    label= TRANSLATIONS[self.current_language]["ai_model"],
                    key="ai_model_dropdown"
                )
            
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        pdf_file = gr.File(
                            label=TRANSLATIONS[self.current_language]["upload_pdf"],
                            file_types=[".pdf"]
                        )
                        with gr.Column():
                            type_model=gr.Radio(choices=["Local", "Api Key"], label=TRANSLATIONS[self.current_language]["model_type"], visible=False, value="Api Key")
                            api_key_input = gr.Textbox(label="Api Key", placeholder=TRANSLATIONS[self.current_language]["api_key_placeholder"], visible=False)
                            project_id_watsonx = gr.Textbox(label="Project ID", placeholder=TRANSLATIONS[self.current_language]["project_id_placeholder"], visible=False)
                    chunk_size = gr.Slider(
                        value=250,
                        label=TRANSLATIONS[self.current_language]["chunk_size"],
                        minimum=100,
                        maximum=500,
                        step=10,
                        visible=False
                    )
                    chunk_overlap = gr.Slider(
                        value=25,
                        label=TRANSLATIONS[self.current_language]["chunk_overlap"],
                        minimum=10,
                        maximum=100,
                        step=5,
                        visible=False
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
                specialist_tab = gr.Tab(TRANSLATIONS[self.current_language]["specialist_tab"])
            with qa_tab:
                chat_title = gr.Markdown(TRANSLATIONS[self.current_language]["chat_title"])
                chat_placeholder = gr.Textbox(
                    placeholder=TRANSLATIONS[self.current_language]["chat_placeholder"],
                    container=False,
                    show_label=False
                )
                chat_btn = gr.Button(TRANSLATIONS[self.current_language]["chat_btn"])
                chatbot = gr.Markdown(height=400)
            
            with summary_tab:
                with gr.Accordion(TRANSLATIONS[self.current_language]["mini_analysis_title"], open=False, visible=False):
                    minisummaries_output = gr.Textbox(
                        label=TRANSLATIONS[self.current_language]["mini_analysis_title"],
                        lines=10
                    )
                summary_output = gr.Textbox(
                    label=TRANSLATIONS[self.current_language]["summary_label"],
                    lines=10
                )
                summarize_btn = gr.Button(
                    TRANSLATIONS[self.current_language]["generate_summary"]
                )
            
            with specialist_tab:
                specialist_title = gr.Markdown(TRANSLATIONS[self.current_language]["specialist_title"])
                specialist_placeholder = gr.Textbox(
                    label=TRANSLATIONS[self.current_language]["specialist_label"],
                    lines=10
                )
                with gr.Accordion(TRANSLATIONS[self.current_language]["mini_analysis_title"], open=False, visible=False):
                    minianalysis_output = gr.Textbox(
                        label=TRANSLATIONS[self.current_language]["mini_analysis_title"],
                        lines=10
                    )
                specialist_output = gr.Textbox(label=TRANSLATIONS[self.current_language]["specialist_output"], lines=20)
                specialist_btn = gr.Button(TRANSLATIONS[self.current_language]["specialist_btn"])

            
            language_dropdown.change(
                fn=self.change_language,
                inputs=[language_dropdown],
                outputs=[
                    title,
                    pdf_file,
                    chunk_size,
                    chunk_overlap,
                    process_btn,
                    process_output,
                    qa_tab,
                    summary_tab,
                    specialist_tab,
                    minisummaries_output,
                    minianalysis_output,
                    chat_placeholder,
                    chat_title,
                    chat_btn,
                    summarize_btn,
                    summary_output,
                    ai_model_dropdown,
                    specialist_title,
                    specialist_placeholder,
                    specialist_output,
                    specialist_btn
                ]
            )

            ai_model_dropdown.change(
                fn=self.change_ai_model,
                inputs=[ai_model_dropdown],
                outputs=[type_model, api_key_input, project_id_watsonx, chunk_size, chunk_overlap]
            )

            type_model.change(
                fn=self.change_type_model,
                inputs=[type_model],
                outputs=[api_key_input,project_id_watsonx]
            )
            
            chat_placeholder.submit(
                fn=self.qa_interface,
                inputs=[chat_placeholder, chatbot, ai_model_dropdown, type_model, api_key_input, project_id_watsonx],
                outputs=[chatbot]
            )
            
            process_btn.click(
                fn=self.process_pdf,
                inputs=[pdf_file, chunk_size, chunk_overlap, ai_model_dropdown, type_model, api_key_input, project_id_watsonx],
                outputs=[process_output]
            )
            
            summarize_btn.click(
                fn=self.summarize_interface,
                inputs=[ai_model_dropdown, type_model, api_key_input, project_id_watsonx],
                outputs=[summary_output]
            )

            specialist_btn.click(
                fn=self.specialist_opinion,
                inputs=[ai_model_dropdown, type_model, api_key_input, project_id_watsonx, specialist_placeholder],
                outputs=[specialist_output]
            )

            chat_btn.click(
                fn=self.qa_interface,
                inputs=[chat_placeholder, chatbot, ai_model_dropdown, type_model, api_key_input, project_id_watsonx],
                outputs=[chatbot]
            )
        
        return demo

if __name__ == "__main__":
    ui = PDFProcessorUI()
    demo = ui.create_ui()
    demo.launch()
