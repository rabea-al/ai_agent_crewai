from crewai import Crew
from agents import TranslatorAgents
from tasks import TranslatorTasks
import os

os.environ["OPENAI_API_KEY"] = ""
class TranslatorCrew:

    def __init__(self, tasks):
        self.tasks = tasks

    def run(self):
        agents = TranslatorAgents()
        task_manager = TranslatorTasks()

        translator_agent = agents.translator_agent()

        crew_tasks = []
        for task in self.tasks:
            if task["type"] == "translation":
                crew_tasks.append(task_manager.translation_task(
                    translator_agent,
                    task["input_file"],
                    task["output_file"],
                    task["target_language"]
                ))
            elif task["type"] == "conversion":
                crew_tasks.append(task_manager.conversion_task(
                    translator_agent,
                    task["input_folder"],
                    task["output_folder"],
                    task["target_format"]
                ))
            elif task["type"] == "financial_analysis":
                crew_tasks.append(task_manager.financial_analysis_task(
                    translator_agent,
                    task["folder_path"],
                    task["text_output_file"],
                    task["json_output_file"]
                    
                ))
            elif task["type"] == "generate_charts":
                crew_tasks.append(task_manager.generate_charts_task(
                    translator_agent,
                    task["input_json"],
                    task["output_folder"]
                ))
            
            else:
                print(f"Unknown task type: {task['type']}")

        crew = Crew(
            agents=[translator_agent],
            tasks=crew_tasks,
            verbose=True
        )

        result = crew.kickoff()
        return result


if __name__ == "__main__":
    tasks = [
        {
            "type": "translation",
            "input_file": "QA.txt",
            "output_file": "translated.txt",
            "target_language": "Japanese"
        },
        {
            "type": "conversion",
            "input_folder": "new_dir",
            "output_folder": "conv",
            "target_format": "png"
        },
        {
            "type": "financial_analysis",
            "folder_path": "financial_docs",
            "text_output_file": "reports/report2024.txt",
            "json_output_file": "reports/report2024.json"
        },
        {
            "type": "generate_charts",
            "input_json": "reports/report2024.json",
            "output_folder": "reports"
        }

       
    ]

    translator_crew = TranslatorCrew(tasks)
    result = translator_crew.run()

    print("\nTask Completed:")
    for res in result:
        print(f"- {res}")
