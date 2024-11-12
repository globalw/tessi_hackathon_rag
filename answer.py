
class Answer:

    def __init__(self, handler, question, answer, billable, context, expectedAnswer, expectedContext):
        self.handler = handler
        self.question = question
        self.answer = answer
        self.billable = billable
        self.context = context
        self.expectedAnswer = expectedAnswer
        self.expectedContext = expectedContext