from pydantic import BaseModel 
class Pergunta(BaseModel): 

#Schema para a requisição de uma pergunta.

    texto: str 

class Resposta(BaseModel): 
    #Schema para a resposta do chatbot. 
    resposta: str
