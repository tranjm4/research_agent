class MemoryModule:
  def __init__(self, capacity=20):
    self.capacity = capacity
    self.current_length = 0

    self.question_trace = []
    self.answer_trace = []
    self.sub_question_trace = []
    self.sub_answer_trace = []
    self.source_trace = []
    self.citation_trace = []

  def add_question(self, question):
    self.question_trace.append(question)
    self.current_length += 1
    if self.current_length > self.capacity:
      self.question_trace.pop(0)
      self.current_length -= 1

  def add_answer(self, answer):
    self.answer_trace.append(answer)
    self.current_length += 1
    if self.current_length > self.capacity:
      self.answer_trace.pop(0)
      self.current_length -= 1

  def add_sub_question(self, sub_question):
    self.sub_question_trace.append(sub_question)

  def add_sub_answer(self, sub_answer):
    self.sub_answer_trace.append(sub_answer)

  def add_source(self, source):
    self.source_trace.append(source)

  def add_citation(self, citation):
    self.citation_trace.append(citation)