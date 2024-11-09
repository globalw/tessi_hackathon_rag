import pandas as pd


class PromptBuilder:
    def __init__(self):
        self.colums = ["customer", "billability", "question", "answer", "sla"]
        self.df = pd.read_csv("test/qa/QA_billability.csv")

    def build_prompts(self):
        prompts = []
        query = ("Write me the following instructions as one string and each of these criterias"
                 " is one field of a row and i dont want to have titles of the criteria just write the answer and sperate each answer with ;: "
                 "'customer name': name of the customer, "
                 "'billable': either 'billable', 'not billable' based on the SLA's, "
                 "'question': just add the query itself, "
                 "'answer': give a response to the customer, "
                 "'sla': add me the indexed sla article index with text. "
                 "these are the queries: "
                 )
        prompt1 = query
        index = 1



        for question in self.df["question"].values:
            if index < 5:
                prompt1 += f"{question}; "
                index += 1
            else:
                prompts.append(prompt1)
                prompt1 = query
                index = 1
        return prompts