from transformers import BartForConditionalGeneration
from transformers import TrainingArguments
from transformers import BartTokenizer
from transformers import Trainer
from datasets import load_dataset

tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
mlsum = load_dataset('mlsum', 'es')


def tokenize(batch):
    model_inputs = tokenizer(batch['text'], max_length=1024, truncation=True,
                             padding='max_length')
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(batch['summary'], max_length=128, truncation=True,
                           padding='max_length')

    model_inputs['labels'] = labels['input_ids']
    return model_inputs


# mlsum_small = mlsum['train'].train_test_split(train_size=0.05, test_size=0.05, seed=42)
# small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
# small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))

tokenized_dataset = mlsum.map(tokenize, batched=True)

small_train_dataset = tokenized_dataset["train"].shuffle(seed=42).select(range(500))
small_eval_dataset = tokenized_dataset["test"].shuffle(seed=42).select(range(500))

training_args = TrainingArguments(
    output_dir='./results',
    eval_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset
)

# Entrenar el modelo
trainer.train()

model.save_pretrained('./modelo_fine_tuned')
tokenizer.save_pretrained('./modelo_fine_tuned')


model = BartForConditionalGeneration.from_pretrained('./modelo_fine_tuned')
tokenizer = BartTokenizer.from_pretrained('./modelo_fine_tuned')

article = """Juan Román Riquelme (San Fernando, 24 de junio de 1978) es un exfutbolista y dirigente deportivo argentino, actual presidente del Club Atlético Boca Juniors.5​ Un emblema del «número 10 clásico», se destacó como uno de los mejores jugadores argentinos de todos los tiempos y más aclamados mediocampistas de su generación, y uno de los últimos referentes de su posición.6​7​8​9​ Además, es ampliamente considerado como el mejor jugador de la historia de Boca Juniors y una de sus más importantes figuras,10​11​12​ debido a su desempeño por 13 temporadas (1996-2014) en el club, en las que consiguió ganar tres Copas Libertadores de América y una Copa Intercontinental, entre otros títulos.
Se formó en las divisiones juveniles de Argentinos Juniors, para más tarde debutar en Boca Juniors en 1996. En el conjunto de la rivera, pasó seis temporadas donde transitó la era más gloriosa del club, ganando tres títulos locales (Apertura 1998, Apertura 2000, Clausura 1999) y tres internacionales (Copas Libertadores 2000, 2001 y Copa Intercontinental 2000). Fue una importante figura entre todos esos campeonatos, potenciado principalmente por su entrenador y mentor Carlos Bianchi. En 2002, fue traspasado al Barcelona, donde tan solo permaneció una temporada por sus problemas con el entrenador del equipo en esa época, Louis van Gaal.13​ Se marchó cedido al Villarreal, club de España donde consiguió sus mayores éxitos en Europa, liderando al equipo a un histórico tercer puesto en la liga y a las semifinales de la Champions League por primera vez en la historia del club.14​15​ En 2007, retornó a Boca y ganó su tercera Copa Libertadores de manera extraordinaria, siendo el goleador del equipo y el mejor jugador del torneo. En su tercer ciclo en el club, consiguió la Recopa Sudamericana 2008 y fue importante en la obtención de los torneos Apertura 2008 y Apertura 2011. Acabó yéndose en 2014, siendo el 6.º jugador con más partidos del club (388), el 7.º con más títulos (11) y su 11.º máximo goleador histórico, con 92 goles. Además es el jugador con más presencias en La Bombonera, con 206. Se retiró en Argentinos Juniors, donde alcanzó el ascenso a la Primera División.
A nivel internacional, fue parte del seleccionado juvenil sub-20 de Argentina, con el cual ganó el Sudamericano Sub-20 de 1997 y el Mundial Juvenil de 1997 disputado en Malasia. En 1997 debutó en la selección mayor, donde tan solo disputó el Mundial de Alemania 2006, quedando afuera en cuartos de final. Alcanzó la final de la Copa Confederaciones 2005 y la Copa América 2007, perdiendo ambas contra Brasil. En 2008, fue parte del equipo que ganó la medalla de oro en los Juegos Olímpicos de Pekín. En 2009 decidió retirarse de la selección, perdiendo la posibilidad de disputar el Mundial de Sudáfrica 2010.
Fue distinguido como el futbolista del año en Argentina en cuatro oportunidades (2000, 2001, 2008 y 2011) y como el futbolista del año en Sudamérica en 2001. Además, obtuvo el Premio Don Balón al mejor jugador extranjero de la Liga Española en la temporada 2004-05, y entre 2005 y 2007 fue incluido en la lista de nominados para las ternas de los premios Jugador Mundial de la FIFA y el Balón de Oro.16​17​ Riquelme fue parte del Equipo Ideal de América en seis oportunidades (1999, 2000, 2001, 2007, 2008 y 2011)."""


def generate_summary(text, model, tokenizer):
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(inputs, max_length=150, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary


summary = generate_summary(article, model, tokenizer)
print("Original Text:", article)
print("Summary:", summary)