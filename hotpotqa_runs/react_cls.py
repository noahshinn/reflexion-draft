import re, string, os
from typing import List
import dotenv
dotenv.load_dotenv()

import tiktoken
from langchain import OpenAI, Wikipedia
from langchain.llms.base import BaseLLM
from langchain.agents.react.base import DocstoreExplorer
from langchain.prompts import PromptTemplate
from prompts import reflect_prompt, react_agent_prompt, react_reflect_agent_prompt, REFLECTION_HEADER
from fewshots import WEBTHINK_SIMPLE6, REFLECTIONS

class ReactAgent:
    def __init__(self,
                 question: str,
                 key: str,
                 max_steps: int = 6,
                 agent_prompt: PromptTemplate = react_agent_prompt,
                 docstore: DocstoreExplorer = DocstoreExplorer(Wikipedia()),
                 react_llm: BaseLLM = OpenAI(
                                            temperature=0,
                                            max_tokens=100,
                                            model_name="text-davinci-003",
                                            model_kwargs={"stop": "\n"},
                                            openai_api_key=os.environ['OPENAI_API_KEY']),
                 ) -> None:
        
        self.question = question
        self.answer = ''
        self.key = key
        self.max_steps = max_steps
        self.agent_prompt = agent_prompt
        self.react_examples = WEBTHINK_SIMPLE6

        self.docstore = docstore # Search, Lookup
        self.llm = react_llm
        
        self.enc = tiktoken.encoding_for_model("text-davinci-003")

        self.__reset_agent()

    def run(self, reset = True) -> None:
        if reset:
            self.__reset_agent()
        
        while not self.is_halted() and not self.is_finished():
            self.step()
    
    def step(self) -> None:
        # Think
        self.scratchpad += f'\nThought {self.step_n}:'
        self.scratchpad += ' ' + self.prompt_agent()
        #print(self.scratchpad.split('\n')[-1])

        # Act
        self.scratchpad += f'\nAction {self.step_n}:'
        action = self.prompt_agent()
        self.scratchpad += ' ' + action
        action_type, argument = parse_action(action)
        #print(self.scratchpad.split('\n')[-1])

        # Observe
        self.scratchpad += f'\nObservation {self.step_n}: '
        
        if action_type == 'Finish':
            self.answer = argument
            if self.is_correct():
                self.scratchpad += 'Answer is CORRECT'
            else: 
                self.scratchpad += 'Answer is INCORRECT'
            self.finished = True
            self.step_n += 1
            return

        if action_type == 'Search':
            try:
                self.scratchpad += format_step(self.docstore.search(argument))
            except Exception as e:
                print(e)
                self.scratchpad += f'Could not find that page, please try again.'
            
        elif action_type == 'Lookup':
            try:
                self.scratchpad += format_step(self.docstore.lookup(argument))
            except ValueError:
                self.scratchpad += f'The last page Searched was not found, so you cannot Lookup a keyword in it. Please try one of the similar pages given.'

        else:
            self.scratchpad += 'Invalid Action. Valid Actions are Lookup[<topic>] Search[<topic>] and Finish[<answer>].'

        #print(self.scratchpad.split('\n')[-1])

        self.step_n += 1

    def prompt_agent(self) -> str:
        return format_step(self.llm(self._build_agent_prompt()))
    
    def _build_agent_prompt(self) -> str:
        return self.agent_prompt.format(
                            examples = self.react_examples,
                            question = self.question,
                            scratchpad = self.scratchpad)
    
    def is_finished(self) -> bool:
        return self.finished

    def is_correct(self) -> bool:
        return EM(self.answer, self.key)

    def is_halted(self) -> bool:
        return ((self.step_n > self.max_steps) or (len(self.enc.encode(self._build_agent_prompt())) > 3896)) and not self.finished

    def __reset_agent(self) -> None:
        self.step_n = 1
        self.finished = False
        self.scratchpad = ''

    def set_qa(self, question: str, key: str) -> None:
        self.question = question
        self.key = key

class ReactReflectAgent(ReactAgent):
    def __init__(self,
                 question: str,
                 key: str,
                 max_steps: int = 6,
                 agent_prompt: PromptTemplate = react_reflect_agent_prompt,
                 reflect_prompt: PromptTemplate = reflect_prompt,
                 docstore: DocstoreExplorer = DocstoreExplorer(Wikipedia()),
                 react_llm: BaseLLM = OpenAI(
                                             temperature=0,
                                             max_tokens=100,
                                             model_name="text-davinci-003",
                                             model_kwargs={"stop": "\n"},
                                             openai_api_key=os.environ['OPENAI_API_KEY']),
                 reflect_llm: BaseLLM = OpenAI(
                                               temperature=0,
                                               max_tokens=250,
                                               model_name="text-davinci-003",
                                               openai_api_key=os.environ['OPENAI_API_KEY']),
                 ) -> None:
        
        super().__init__(question, key, max_steps, agent_prompt, docstore, react_llm)
        self.reflect_llm = reflect_llm
        self.reflect_prompt = reflect_prompt
        self.reflect_examples = REFLECTIONS
        self.reflections = []
    
    def run(self, reset = True) -> None:
        if (self.is_finished() or self.is_halted()) and not self.is_correct():
            self.reflect()

        ReactAgent.run(self, reset)
    
    def reflect(self) -> None:
        self.reflections.append(self.prompt_reflection())
    
    def prompt_reflection(self) -> str:
        return format_step(self.reflect_llm(self._build_reflection_prompt()))

    def _build_reflection_prompt(self) -> str:
        return self.reflect_prompt.format(
                            examples = self.reflect_examples,
                            question = self.question,
                            scratchpad = self._format_scratchpad())
 
    def _build_agent_prompt(self) -> str:
        return self.agent_prompt.format(
                            examples = self.react_examples,
                            reflections = format_reflections(self.reflections),
                            question = self.question,
                            scratchpad = self.scratchpad)
    
    def _format_scratchpad(self) -> str:
        lines = self.scratchpad.split('\n')
        lines_by_tokens = sorted(lines, key=lambda x: len(self.enc.encode(x)))
        while len(self.enc.encode('\n'.join(lines))) > 1600:
            lines.remove(lines_by_tokens.pop(-1))
        return '\n'.join(lines)
    

### String Stuff ###

def parse_action(string):
    pattern = r'^(\w+)\[(.+)\]$'
    match = re.match(pattern, string)
    
    if match:
        action_type = match.group(1)
        argument = match.group(2)
        return action_type, argument
    
    else:
        return None

def format_step(step: str) -> str:
    return step.strip('\n').strip().replace('\n', '')

def format_reflections(reflections: List[str]) -> str:
    if reflections == []:
        return ''
    else:
        header = REFLECTION_HEADER
        return header + 'Reflections:\n- ' + '\n- '.join([r.strip() for r in reflections])

def normalize_answer(s):
  def remove_articles(text):
    return re.sub(r"\b(a|an|the)\b", " ", text)
  
  def white_space_fix(text):
      return " ".join(text.split())

  def remove_punc(text):
      exclude = set(string.punctuation)
      return "".join(ch for ch in text if ch not in exclude)

  def lower(text):
      return text.lower()

  return white_space_fix(remove_articles(remove_punc(lower(s))))

def EM(answer, key) -> bool:
    return normalize_answer(answer) == normalize_answer(key)



