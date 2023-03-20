from typing import List
from react_cls import format_reflections, ReactAgent

def summarize_trial(agents):
    correct = [a for a in agents if a.is_correct()]
    halted = [a for a in agents if a.is_halted()]
    incorrect = [a for a in agents if a.is_finished() and not a.is_correct()]
    return correct, incorrect, halted

    # (Not correct) and (not halted) and (not finished and correct)
def log_trial(agents: List[ReactAgent], trial_n):
    correct, incorrect, halted = summarize_trial(agents)

    log = f"""
########################################
BEGIN TRIAL {trial_n}
Trial summary: Correct: {len(correct)}, Incorrect: {len(incorrect)}, Halted: {len(halted)}
#######################################
"""

    log += '------------- BEGIN CORRECT AGENTS -------------\n\n'
    for agent in correct:
        log += f'Question: {agent.question}{format_reflections(agent.reflections)}{agent.scratchpad}\nCorrect answer: {agent.key}\n\n'

    log += '------------- BEGIN INCORRECT AGENTS -----------\n\n'
    for agent in incorrect:
        log += f'Question: {agent.question}{format_reflections(agent.reflections)}{agent.scratchpad}\nCorrect answer: {agent.key}\n\n'

    log += '------------- BEGIN HALTED AGENTS --------------\n\n'
    for agent in halted:
        log += f'Question: {agent.question}{format_reflections(agent.reflections)}{agent.scratchpad}\nCorrect answer: {agent.key}\n\n'

    return log