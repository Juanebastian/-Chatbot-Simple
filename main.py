from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer
from deep_translator import GoogleTranslator

# Crear una instancia de ChatBot (sigue en inglés)
chatbot = ChatBot(
    'SimpleBot',
    storage_adapter='chatterbot.storage.SQLStorageAdapter',
    logic_adapters=[
        'chatterbot.logic.BestMatch',
        'chatterbot.logic.MathematicalEvaluation'
    ],
    database_uri='sqlite:///database.db'  # Usando la base de datos en inglés
)

# Entrenador para el ChatBot (sigue en inglés)
trainer = ChatterBotCorpusTrainer(chatbot)

# Entrenando el chatbot con el corpus de conversación en inglés
trainer.train('chatterbot.corpus.english')


# Función para interactuar con el bot con traducción
def chat():
    print("¡Hola! Soy un chatbot con traducción automática. Puedes escribirme en español.")
    print("Para salir escribe 'salir'.")

    # Creamos los traductores
    es_to_en = GoogleTranslator(source='es', target='en')
    en_to_es = GoogleTranslator(source='en', target='es')

    while True:
        try:
            # Obtener la entrada del usuario en español
            user_input_es = input("Tú (español): ")

            if user_input_es.lower() == 'salir':
                print("Adiós, ¡hasta luego!")
                break

            # Traducir la entrada del usuario al inglés
            user_input_en = es_to_en.translate(user_input_es)
            print(f"[Traducido al inglés: {user_input_en}]")

            # Obtener la respuesta del bot (en inglés)
            bot_response_en = str(chatbot.get_response(user_input_en))
            print(f"[Bot en inglés: {bot_response_en}]")

            # Traducir la respuesta del bot al español
            bot_response_es = en_to_es.translate(bot_response_en)
            print(f"Bot (español): {bot_response_es}")

        except (KeyboardInterrupt, EOFError, SystemExit):
            break
        except Exception as e:
            print(f"Error: {e}")
            print("Intenta de nuevo con una frase diferente.")


if __name__ == "__main__":
    chat()