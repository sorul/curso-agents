from agents import Runner, trace, gen_trace_id
from clarifier_agent import clarifier_agent, ClarifyingQuestions
from search_agent import search_agent
from planner_agent import planner_agent, WebSearchItem, WebSearchPlan
from writer_agent import writer_agent, ReportData
from email_agent import email_agent
import asyncio


class ResearchManager:

    async def clarify_query(self, query: str) -> ClarifyingQuestions:
        """ Genera preguntas de aclaración antes de planificar la investigación """
        print("Generando preguntas de aclaración...")
        result = await Runner.run(
            clarifier_agent,
            f"Consulta inicial: {query}",
        )
        return result.final_output_as(ClarifyingQuestions)

    async def run(self, query: str, clarification_answers: str):
        """ Ejecuta el proceso de investigación profunda, generando los actualizaciones de estado y el informe final """
        trace_id = gen_trace_id()
        with trace("Ingestigación", trace_id=trace_id):
            print(
                f"Ver traza: https://platform.openai.com/traces/trace?trace_id={trace_id}"
            )
            yield f"Ver traza: https://platform.openai.com/traces/trace?trace_id={trace_id}"

            print("Iniciando investigación...")
            research_brief = self.build_research_brief(query, clarification_answers)

            # planner_agent.py
            search_plan = await self.plan_searches(research_brief)
            yield "Búsquedas planificadas, iniciando búsqueda..."

            # search_agent.py (se puede paralelizar cada búsqueda)
            search_results = await self.perform_searches(search_plan)
            yield "Búsquedas completas, escribiendo informe..."

            # writer_agent.py
            report = await self.write_report(research_brief, search_results)
            yield "Informe escrito, enviando correo electrónico..."

            # email_agent.py
            await self.send_email(report)
            yield "Correo electrónico enviado, investigación completa"

            yield report.markdown_report

    def build_research_brief(self, query: str, clarification_answers: str) -> str:
        """ Construye el contexto que usarán los agentes posteriores """
        answers = clarification_answers.strip() or "El usuario no proporcionó aclaraciones adicionales."
        return f"""Consulta inicial:
{query}

Aclaraciones del usuario:
{answers}
"""

    async def plan_searches(self, research_brief: str) -> WebSearchPlan:
        """ Planifica las búsquedas a realizar para la consulta """
        print("Planificando búsquedas...")
        result = await Runner.run(
            planner_agent,
            f"Brief de investigación:\n{research_brief}",
        )
        print(f"Se realizarán {len(result.final_output.searches)} búsquedas")
        return result.final_output_as(WebSearchPlan)

    async def perform_searches(self, search_plan: WebSearchPlan) -> list[str]:
        """ Realiza las búsquedas para la consulta """
        print("Buscando...")
        num_completed = 0
        tasks = [
            asyncio.create_task(self.search(item))
            for item in search_plan.searches
        ]
        results = []
        for task in asyncio.as_completed(tasks):
            result = await task
            if result is not None:
                results.append(result)
            num_completed += 1
            print(f"Buscando... {num_completed}/{len(tasks)} completadas")
        print("Búsqueda completada")
        return results

    async def search(self, item: WebSearchItem) -> str | None:
        """ Realiza una búsqueda para la consulta """
        user_prompt = f"Término de búsqueda: {item.query}\nRazón para buscar: {item.reason}"
        try:
            result = await Runner.run(
                search_agent,
                user_prompt,
            )
            return str(result.final_output)
        except Exception:
            return None

    async def write_report(self, research_brief: str,
                           search_results: list[str]) -> ReportData:
        """ Escribe el informe para la consulta """
        print("Pensando en el informe...")
        user_prompt = f"Brief de investigación: {research_brief}\nResultados de búsqueda resumidos: {search_results}"
        result = await Runner.run(
            writer_agent,
            user_prompt,
        )

        print("Informe escrito")
        return result.final_output_as(ReportData)

    async def send_email(self, report: ReportData) -> None:
        print("Escribiendo correo electrónico...")
        result = await Runner.run(
            email_agent,
            report.markdown_report,
        )
        print("Correo electrónico enviado")
        return report
