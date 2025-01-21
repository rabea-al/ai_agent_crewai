from crewai import Task
from textwrap import dedent


class TranslatorTasks:

    def translation_task(self, agent, input_file, output_file, target_language):
        return Task(
            description=dedent(f"""
                Translate the text from {input_file} to {target_language} and save it to {output_file}.
            """),
            agent=agent,
            expected_output=f"Translated content saved in {output_file}"
        )

    def conversion_task(self, agent, input_folder, output_folder, target_format):
        return Task(
            description=dedent(f"""
                Convert all images in {input_folder} to {target_format} format and save them in {output_folder}.
            """),
            agent=agent,
            expected_output=f"All images converted to {target_format.upper()} and saved in {output_folder}"
        )

    def financial_analysis_task(self, agent, folder_path, text_output_file, json_output_file):
        return Task(
            description=f"Analyze financial data in {folder_path} and generate two reports: {text_output_file} (text) and {json_output_file} (JSON).",
            agent=agent,
            expected_output=f"Reports saved in {text_output_file} and {json_output_file}",
            inputs={
                "folder_path": folder_path,
                "text_output_file": text_output_file,
                "json_output_file": json_output_file
            }
        )
    def generate_charts_task(self, agent, input_json, output_folder):
        return Task(
            description=f"Generate charts from the financial analysis JSON report: {input_json}. Save charts to {output_folder}.",
            agent=agent,
            expected_output=f"Charts saved to {output_folder}",
            inputs={
                "input_json": input_json,
                "output_folder": output_folder
            }
        )

   

