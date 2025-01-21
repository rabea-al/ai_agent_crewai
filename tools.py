from PIL import Image
import os
from langchain_community.tools import tool
import deepl
import pandas as pd
import json
import matplotlib.pyplot as plt


class ConversionTools:
    name = "ConversionTools"
    description = "converts a batch of images from one format to another and saves the converted images in a specified output directory."
    
    @tool("Convert images")
    def convert_images(input_folder, output_folder, target_format):
        """
        Convert all images in input_folder to target_format and save in output_folder.
        """
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        for filename in os.listdir(input_folder):
            if filename.lower().endswith(("png", "jpg", "jpeg", "bmp", "gif")):
                input_path = os.path.join(input_folder, filename)
                base_name = os.path.splitext(filename)[0]
                output_path = os.path.join(output_folder, f"{base_name}.{target_format}")

                try:
                    with Image.open(input_path) as img:
                        img.convert("RGB").save(output_path, target_format.upper())
                except Exception as e:
                    print(f"Error converting {filename}: {str(e)}")

        return f"All images converted to {target_format.upper()} and saved in {output_folder}"



class TranslationTools:
    name = "TranslationTools"
    description = " Translate all texts within files You must send the language code (e.g., 'en' for English, 'ar' for Arabic) in `target_language` instead of the full language name."
    os.environ["DEEPL_API_KEY"] =""
    @tool("Translate text")
    def translate_text(input_file, target_language, output_file):
        """
        Translate the text from input_file to target_language using DeepL 
        and save to output_file.
        """
        with open(input_file, 'r', encoding='utf-8') as f:
            text = f.read()

        auth_key = os.getenv("DEEPL_API_KEY") 
        if not auth_key:
            return "DeepL API key is missing. Please set the DEEPL_API_KEY environment variable."

        translator = deepl.Translator(auth_key)

        try:
            translated_text = translator.translate_text(text, target_lang=target_language.upper())
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(translated_text.text)

            return f"Translation completed. Saved to {output_file}"
        except Exception as e:
            return f"An error occurred during translation: {str(e)}"




class FinancialAnalysisTools:
    name = "FinancialAnalysisTools"
    description = "Reads financial data from documents and generates a report saved to a specified text file."

    @tool("Analyze financial data")
    def analyze_financial_data(folder_path, text_output_file, json_output_file):
        """
        Read financial data and generate a report, save it in the specified output file.
        """
        def read_financial_documents(folder_path):
            data_frames = []
            for file_name in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file_name)
                if file_name.endswith('.csv'):
                    df = pd.read_csv(file_path)
                elif file_name.endswith('.xlsx'):
                    df = pd.read_excel(file_path)
                else:
                    print(f"Skipping unsupported file: {file_name}")
                    continue
                data_frames.append(df)
            return pd.concat(data_frames, ignore_index=True)

        def analyze_data(data):
            analysis = {
                'total_revenue': data['Revenue'].sum(),
                'total_expenses': data['Expenses'].sum(),
                'net_profit': data['Revenue'].sum() - data['Expenses'].sum(),
                'average_revenue': data['Revenue'].mean(),
                'average_expenses': data['Expenses'].mean(),
                'average_customer_satisfaction': data['Customer Satisfaction (%)'].mean(),
            }

            department_analysis = {}
            for department in data['Department'].unique():
                dept_data = data[data['Department'] == department]
                department_analysis[department] = {
                    'total_revenue': dept_data['Revenue'].sum(),
                    'total_expenses': dept_data['Expenses'].sum(),
                    'net_profit': dept_data['Revenue'].sum() - dept_data['Expenses'].sum(),
                }
            analysis['department_analysis'] = department_analysis

            region_analysis = {}
            for region in data['Region'].unique():
                region_data = data[data['Region'] == region]
                region_analysis[region] = {
                    'total_revenue': region_data['Revenue'].sum(),
                    'total_expenses': region_data['Expenses'].sum(),
                    'net_profit': region_data['Revenue'].sum() - region_data['Expenses'].sum(),
                }
            analysis['region_analysis'] = region_analysis

            analysis['total_transactions'] = data['Number of Transactions'].sum()
            analysis['average_transactions'] = data['Number of Transactions'].mean()

            return analysis

        try:
            output_folder = os.path.dirname(text_output_file)
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
                print(f"Folder '{output_folder}' created successfully.")
                
            data = read_financial_documents(folder_path)
            analysis = analyze_data(data)

            with open(text_output_file, 'w', encoding='utf-8') as f:
                f.write(f"Total Revenue: {analysis['total_revenue']}\n")
                f.write(f"Total Expenses: {analysis['total_expenses']}\n")
                f.write(f"Net Profit: {analysis['net_profit']}\n")
                f.write(f"Average Revenue: {analysis['average_revenue']:.2f}\n")
                f.write(f"Average Expenses: {analysis['average_expenses']:.2f}\n")
                f.write(f"Average Customer Satisfaction: {analysis['average_customer_satisfaction']:.4f}\n\n")

                f.write("Department Analysis:\n")
                for dept, stats in analysis['department_analysis'].items():
                    f.write(f"- {dept}: Total Revenue: {stats['total_revenue']}, Total Expenses: {stats['total_expenses']}, Net Profit: {stats['net_profit']}\n")

                f.write("\nRegion Analysis:\n")
                for region, stats in analysis['region_analysis'].items():
                    f.write(f"- {region}: Total Revenue: {stats['total_revenue']}, Total Expenses: {stats['total_expenses']}, Net Profit: {stats['net_profit']}\n")

                f.write(f"\nTotal Transactions: {analysis['total_transactions']}\n")
                f.write(f"Average Transactions: {analysis['average_transactions']:.1f}\n")

            with open(json_output_file, 'w', encoding='utf-8') as f:
                json.dump(
                    {k: (v if isinstance(v, (str, float, int)) else str(v)) for k, v in analysis.items()},
                    f,
                    indent=4
                )
            return f"Analysis completed. Text report saved to {text_output_file}, and JSON report saved to {json_output_file}"
        except Exception as e:
            return f"An error occurred during analysis: {str(e)}"

class VisualizationTools:
    name = "VisualizationTools"
    description = "Generate charts from JSON financial data."

    @tool
    def generate_charts(input_json, output_folder):
        """
        Generate three charts based on financial data from JSON file:
        - total_revenue_vs_expenses
        - regional_performance
        - net_profit_by_department
        """
        try:
            with open(input_json, 'r', encoding='utf-8') as f:
                data = json.load(f)

            department_analysis = eval(data["department_analysis"])
            region_analysis = eval(data["region_analysis"])

            department_analysis = eval(data["department_analysis"])
            region_analysis = eval(data["region_analysis"])

            plt.figure(figsize=(8, 6))
            revenue = int(data["total_revenue"])
            expenses = int(data["total_expenses"])
            labels = ['Total Revenue', 'Total Expenses']
            values = [revenue, expenses]
            plt.bar(labels, values, color=['green', 'red'])
            plt.title('Total Revenue vs Total Expenses')
            plt.ylabel('Amount')
            plt.savefig(f"{output_folder}/total_revenue_vs_expenses.png")
            plt.close()

            plt.figure(figsize=(10, 6))
            regions = list(region_analysis.keys())
            revenues = [region_analysis[region]['total_revenue'] for region in regions]
            expenses = [region_analysis[region]['total_expenses'] for region in regions]
            x = range(len(regions))
            plt.bar(x, revenues, width=0.4, label='Revenue', align='center', color='blue')
            plt.bar(x, expenses, width=0.4, label='Expenses', align='edge', color='orange')
            plt.xticks(x, regions)
            plt.title('Regional Performance')
            plt.xlabel('Regions')
            plt.ylabel('Amount')
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"{output_folder}/regional_performance.png")
            plt.close()

            plt.figure(figsize=(10, 6))
            departments = list(department_analysis.keys())
            net_profits = [department_analysis[dept]['net_profit'] for dept in departments]
            plt.bar(departments, net_profits, color='purple')
            plt.title('Net Profit by Department')
            plt.xlabel('Departments')
            plt.ylabel('Net Profit')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f"{output_folder}/net_profit_by_department.png")
            plt.close()

            return f"Charts generated and saved to {output_folder}"
        except Exception as e:
            return f"An error occurred during chart generation: {str(e)}"
