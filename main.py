from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer



# Crear una instancia de ChatBot

chatbot = ChatBot(
    'SimpleBot',
    storage_adapter='chatterbot.storage.SQLStorageAdapter',
    logic_adapters=[
        'chatterbot.logic.BestMatch',
        'chatterbot.logic.MathematicalEvaluation'
    ],
    database_uri='sqlite:///database.db'  # Usando una base de datos SQLite
)

# Entrenador para el ChatBot
trainer = ChatterBotCorpusTrainer(chatbot)

# Entrenando el chatbot con el corpus de conversación en inglés
trainer.train('chatterbot.corpus.english')

# Función para interactuar con el bot
def chat():
    print("¡Hola! Soy un chatbot. ¿En qué puedo ayudarte?")
    while True:
        try:
            # Obtener la entrada del usuario
            user_input = input("Tú: ")

            if user_input.lower() == 'salir':
                print("Adiós, ¡hasta luego!")
                break

            # Obtener la respuesta del bot
            bot_response = chatbot.get_response(user_input)
            print(f"Bot: {bot_response}")
        except (KeyboardInterrupt, EOFError, SystemExit):
            break

if __name__ == "__main__":
    chat()
