from transformers import pipeline, set_seed
import random
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Descargar recursos necesarios de NLTK (solo la primera vez)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')


class ChatbotAvanzado:
    def __init__(self):
        print("Inicializando chatbot avanzado...")

        # Cargar el generador de texto
        try:
            self.generator = pipeline('text-generation', model='gpt2-medium')
            set_seed(42)  # Para resultados reproducibles
            self.use_generator = True
            print("Modelo de generación cargado correctamente.")
        except Exception as e:
            print(f"No se pudo cargar el modelo de generación: {e}")
            print("El chatbot funcionará en modo básico.")
            self.use_generator = False

        # Inicializar procesador de lenguaje natural
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

        # Base de conocimiento simple para respuestas predefinidas
        self.knowledge_base = {
            'saludo': [
                "¡Hola! ¿Cómo estás hoy?",
                "¡Saludos! ¿En qué puedo ayudarte?",
                "¡Buen día! Estoy aquí para asistirte."
            ],
            'despedida': [
                "¡Hasta luego! Espero verte pronto.",
                "Adiós, fue un placer hablar contigo.",
                "Que tengas un buen día. ¡Vuelve cuando quieras!"
            ],
            'gracias': [
                "De nada, estoy para ayudarte.",
                "No hay de qué. ¿Hay algo más en lo que pueda asistirte?",
                "Es un placer poder ser útil."
            ],
            'nombre': [
                "Soy un chatbot avanzado local. Puedes llamarme Asistente.",
                "Me llamo Asistente, soy un modelo de chatbot que funciona localmente en tu computadora.",
                "Soy Asistente, un chatbot diseñado para conversar y ayudarte con información."
            ],
            'capacidades': [
                "Puedo mantener conversaciones, responder preguntas básicas y generar texto basado en lo que me escribas.",
                "Estoy diseñado para interactuar de forma conversacional y ayudarte con información básica.",
                "Puedo conversar sobre temas generales, aunque mis conocimientos son limitados comparados con modelos más grandes."
            ],
            'tiempo': [
                "No tengo acceso a información en tiempo real como el clima o la fecha actual.",
                "No puedo acceder a internet para ver el clima o noticias actuales.",
                "Soy un modelo local, así que no puedo consultar información externa como el clima."
            ],
            'aprendizaje': [
                "Soy un modelo pre-entrenado. No aprendo de nuestras conversaciones.",
                "No tengo la capacidad de aprender o recordar conversaciones anteriores después de reiniciarme.",
                "Mi modelo es estático y no se actualiza con nuevos datos durante nuestras conversaciones."
            ],
            'razonamiento': [
                "Analicemos este problema paso a paso para encontrar una solución lógica.",
                "Vamos a pensar en esto de manera sistemática para llegar a una conclusión.",
                "Podemos abordar este tema descomponiéndolo en partes más pequeñas y manejables."
            ],
            'desconocido': [
                "No estoy seguro de entender completamente. ¿Podrías darme más detalles?",
                "Ese es un tema interesante. ¿Podrías elaborar un poco más?",
                "No tengo suficiente contexto para responder adecuadamente. ¿Podrías reformular tu pregunta?"
            ]
        }

        # Patrones para identificar intenciones
        self.patterns = {
            'saludo': r'(hola|saludos|buenos\s(días|tardes|noches)|qué\stal|hey|hi)',
            'despedida': r'(adiós|chao|hasta\sluego|nos\svemos|bye|salir|exit)',
            'gracias': r'(gracias|te\slo\sagradezco|thanks)',
            'nombre': r'((cómo\ste\sllamas)|(cuál\ses\stu\snombre)|(quién\seres)|(qué\seres))',
            'capacidades': r'(qué\spuedes\shacer|cuáles\sson\stus\scapacidades|para\squé\ssirves)',
            'tiempo': r'(qué\stimpo\shace|cuál\ses\sel\sclima|temperatura)',
            'aprendizaje': r'(puedes\saprender|aprendes|memorizas)',
            'razonamiento': r'(razona|analiza|piensa|resuelve|soluciona)'
        }

        # Mantener contexto de la conversación
        self.conversation_history = []
        self.max_history = 5

        print("Chatbot listo para conversar.")

    def preprocess_text(self, text):
        # Convertir a minúsculas
        text = text.lower()

        # Tokenizar
        tokens = word_tokenize(text)

        # Eliminar palabras vacías y lematizar
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens if word.isalnum() and word not in self.stop_words]

        return tokens

    def identify_intent(self, text):
        text = text.lower()

        # Verificar patrones conocidos
        for intent, pattern in self.patterns.items():
            if re.search(pattern, text):
                return intent

        return 'desconocido'

    def get_response_from_kb(self, intent):
        if intent in self.knowledge_base:
            return random.choice(self.knowledge_base[intent])
        return random.choice(self.knowledge_base['desconocido'])

    def generate_advanced_response(self, text):
        # Combinar el historial de conversación con la entrada actual
        context = " ".join(self.conversation_history[-3:] + [text])

        # Generar texto
        try:
            response = self.generator(context, max_length=100, num_return_sequences=1)[0]['generated_text']

            # Extraer solo la parte nueva (respuesta del chatbot)
            new_text = response[len(context):].strip()

            # Limpiar la respuesta
            new_text = re.split(r'[.!?]\s', new_text)[0]  # Tomar solo la primera oración

            if len(new_text) < 5:  # Si la respuesta es muy corta, usar respuesta predefinida
                return self.get_response_from_kb('desconocido')

            return new_text + "."
        except Exception:
            return self.get_response_from_kb('desconocido')

    def update_conversation_history(self, user_input, bot_response):
        self.conversation_history.append(user_input)
        self.conversation_history.append(bot_response)

        # Mantener el historial con un tamaño máximo
        if len(self.conversation_history) > self.max_history * 2:
            self.conversation_history = self.conversation_history[-self.max_history * 2:]

    def chat(self):
        print("¡Hola! Soy un chatbot avanzado con capacidad de procesamiento de lenguaje natural.")
        print("Puedes conversar conmigo sobre diversos temas.")
        print("Para salir escribe 'salir', 'adios' o 'exit'.")

        while True:
            user_input = input("\nTú: ")

            # Verificar si el usuario quiere salir
            if re.search(self.patterns['despedida'], user_input.lower()):
                print(random.choice(self.knowledge_base['despedida']))
                break

            # Identificar la intención del usuario
            intent = self.identify_intent(user_input)

            # Generar respuesta
            if self.use_generator and random.random() > 0.5 and intent == 'desconocido':
                response = self.generate_advanced_response(user_input)
            else:
                response = self.get_response_from_kb(intent)

            print(f"Bot: {response}")

            # Actualizar historial
            self.update_conversation_history(user_input, response)


# Ejecutar el chatbot
if __name__ == "__main__":
    chatbot = ChatbotAvanzado()
    chatbot.chat()