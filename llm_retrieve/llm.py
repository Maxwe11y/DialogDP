
import os
import openai
from typing import List
openai.organization = "org"
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.Model.list()


def get_prompt(u_i: str,
    rel_head: List[str],
    head_utts: List[str],
    rel_dep: List[str]=None,
    dep_utts: List[str]=None) -> str:

    deps_head = "".join(["Relation: {}, Head: {},".format(r, h) for r, h in zip(rel_head, head_utts)])
    if rel_dep and dep_utts:
        deps_dep = "".join(["Relation: {}, Dependent: {},".format(r, d) for r, d in zip(rel_dep, dep_utts)])
    else:
        deps_dep = ""
    deps = deps_head + deps_dep
    prompt = "Current utterance: {}, {}, Emotion label for current utterance is:".format(u_i,deps)
    return prompt


def gen_response(prompt, task="sentiment analysis", engine="gpt-3.5-turbo-0301"):
    if task == "sentiment analysis":
        sys_content = "You are an expert in sentiment analysis." \
                      + "Please predict the emotion labels for the current utterance given its head and dependent utterances and the relation types."\
                      + "The emotion label is chosen from [Neutral, Sad, Frustrated, Happy, Excited, Angry]."
    elif task == "emotion cause extraction":
        sys_content = "You are an expert in emotion cause extraction." \
                      + " Given dialogue history, current utterance's head utterances and the relation types, please predict whether the candidate utterance is the emotiona cause of the current utterance." \
                      + " The answer should be one from [Yes, No]."
    else:
        sys_content = ""

    completion = openai.ChatCompletion.create(
        model= engine,
        messages=[
        {"role": "system", "content": sys_content},
        {"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=256,
        frequency_penalty=0,
        presence_penalty=0
    )

    print(completion.choices[0].message)

