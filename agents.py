from crewai import Agent
from tools import TranslationTools, ConversionTools, FinancialAnalysisTools, VisualizationTools

class TranslatorAgents:

    def translator_agent(self):
        return Agent(
            role='Translator, Image Converter, Financial Analyst, and Chart Generator',
            goal='Translate texts, convert images, analyze financial data, and generate financial charts.',
            backstory='A multi-functional agent capable of handling various tasks.',
            tools=[
                TranslationTools.translate_text,
                ConversionTools.convert_images,
                FinancialAnalysisTools.analyze_financial_data,
                VisualizationTools.generate_charts,
            ],
            verbose=True
        )
