from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from dotenv import load_dotenv

load_dotenv()

@CrewBase
class VideoAnnotations():
	"""VideoAnnotations crew"""

	agents_config = 'config/agents.yaml'
	tasks_config = 'config/tasks.yaml'

	@agent
	def AnnotationConverterAgent(self) -> Agent:
		return Agent(
			config=self.agents_config['AnnotationConverterAgent'],
			verbose=True
		)
	
	@agent
	def DataQualityAgent(self) -> Agent:
		return Agent(
			config=self.agents_config['DataQualityAgent'],
			verbose=True
		)

	@task
	def ConvertAnnotationsToCSV(self) -> Task:
		return Task(
			config=self.tasks_config['ConvertAnnotationsToCSV'],
			output_file='result.md'
		)
	
	@task
	def ValidateCSVStructure(self) -> Task:
		return Task(
			config=self.tasks_config['ValidateCSVStructure'],
		)
	@task
	def GenerateFinalCSV(self) -> Task:
		return Task(
			config=self.tasks_config['GenerateFinalCSV'],
			# output_file='result.md'
		)

	@crew
	def crew(self) -> Crew:
		# To learn how to add knowledge sources to your crew, check out the documentation:
		# https://docs.crewai.com/concepts/knowledge#what-is-knowledge

		return Crew(
			agents=self.agents, # Automatically created by the @agent decorator
			tasks=self.tasks, # Automatically created by the @task decorator
			process=Process.sequential,
			verbose=True,
			# process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
		)
