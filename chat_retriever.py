import config
from langchain.vectorstores.chroma import Chroma
from langchain import PromptTemplate
from langchain.llms import AzureOpenAI
from langchain.chains import RetrievalQA
from operator import itemgetter
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
import os
from googletrans import Translator

translator = Translator()

os.environ["OPENAI_API_TYPE"] = config.OPENAI_API_TYPE
os.environ["OPENAI_API_VERSION"] = config.OPENAI_API_VERSION
os.environ["OPENAI_API_BASE"] = config.OPENAI_API_BASE
os.environ["OPENAI_API_KEY"] = config.OPENAI_API_KEY
api_key = os.getenv("OPENAI_API_KEY")

class question_answering():

    def __init__(self):
        embedding_hf = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
        db = Chroma(persist_directory= os.getcwd() + config.CHROMA_PATH, embedding_function=embedding_hf)

        self.embedding_hf = embedding_hf
        self.db = db

    def QA_Retriever(self,question:str) -> str :

        if config.TRANSLATE_ANSWER:
            self.question = translator.translate(question, to_lang='en', dest='en').text
        else:
            self.question = question

        template = """
        Answer the question based only on the following context and paraphrase, but if User want to analysis or suggest somw idea, please helpful:
        {context}

        Question: {question}
        Answer: 
        """
        prompt = ChatPromptTemplate.from_template(template)

        memory = ConversationBufferMemory(return_messages=True)

        model = AzureOpenAI(
            engine="nong-model",
            model_name="gpt-4",
            temperature=0.7,
        )

        retriever = self.db.as_retriever(k=config.NUM_RETRIEVER)
        chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | RunnablePassthrough.assign(
                history=RunnableLambda(memory.load_memory_variables) | itemgetter("history"))
            | prompt
            | model
            | StrOutputParser()
        )
        text_reply = chain.invoke(self.question)
                                  
        if config.TRANSLATE_ANSWER:
            return translator.translate(text_reply, to_lang='th', dest='th').text
        else:
            return text_reply

    def get_relevant(self) -> set:
        sources = self.db.similarity_search(self.question, k=config.NUM_RETRIEVER)
        source_ls = []
        data_ls = []
        for i in sources:
            if config.TRANSLATE_ANSWER:
                source_ls.append(i.metadata['source'])
                data_ls.append(translator.translate(i.page_content[:1500], to_lang='th', dest='th').text)
                source_send = source_ls
                data_send = data_ls
            else:
                source_ls.append([i.metadata['source'],i.page_content])
                data_ls.append(i.page_content)
        return set(source_ls),set(data_ls)




