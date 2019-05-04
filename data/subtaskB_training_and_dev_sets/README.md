===============================================================================

NEGES 2019 task - Negation in Spanish - Subtask B: Role of negation in sentiment analysis
 
Training and development sets released: February 21, 2019.

Web site: http://www.sepln.org/workshops/neges2019/

------------------------------------
Organizing comittee
------------------------------------
Salud María Jiménez Zafra, sjzafra@ujaen.es (Universidad de Jaén, Spain)
Maite Martín Valdivia, maite@ujaen.es (Universidad de Jaén, Spain)
Noa Cruz Díaz, contact@noacruz.com (Savana Médica, Madrid, Spain)
Roser Morante, r.morantevallejo@vu.nl (VU Amsterdam, Netherlands)
===============================================================================

------------------------------------
CONTENT FOLDER
------------------------------------
The folder "subtaskB" contains the training and development sets for subtask B and the evaluation script.
The subfolder "corpus_SFU_Review_SP_NEG_subtaskB" contains the corpus and the gold labels.
The subfolder "scorer" contains the evaluation script.


------------------------------------
ABOUT THE TASK
------------------------------------
Negation is a complex linguistic phenomenon of growing interest in computational linguistics. Detection and treatment of negation is relevant in a wide range of applications, such as information extraction, machine translation or sentiment analysis, where it is crucial to know when a part of the text should have a different meaning due to the presence of negation. In recent years, several
challenges and shared tasks have focused on processing negation: NeSp-NLP 2010 (Morante and Sporleder, 2010), CoNLL-2010 share task (Farkas et al., 2010) and SEM 2012 shared task (Morante
and Blanco, 2012). However, most of the research on negation has been done for English. Therefore, this task aims to advance the study of this phenomenon in Spanish, the second most
widely spoken language in the world and the third most widely used on the Internet. The main objective is to bring together the scientific community that is working on this issue to discuss how this
phenomenon is being addressed, what are the main problems encountered, as well as sharing resources and tools aimed at negation in Spanish.

This task was also organized last year as a workshop, NEGES: Workshop on Negation in Spanish (Jiménez-Zafra et al., 2018a; Jiménez-Zafra et al., 2018b), held as part of the XXXIV edition of the
International Conference of the Spanish Society for Natural Language Processing (SEPLN 2018). This year, it is presented in IberLEF (Iberian Languages Evaluation Forum), collocated with SEPLN
2019 Conference, with the aim of joining forces with other researchers to create a reference forum in Spanish with tasks of relevance to processing some of the languages spoken in the Iberian
Peninsula.


------------------------------------
SUBTASK DESCRIPTION
------------------------------------
Subtask B: Role of negation in sentiment analysis

It is a specific task whose objective is to evaluate the role of negation in sentiment analysis. In this case, participants should develop a system that uses the negation information contained in the SFU ReviewSP-NEG corpus (Jiménez-Zafra et al., 2018c) to improve the task of polarity classification. Systems will have to classify each review as positive or negative using a negation processing
heuristic. The SFU ReviewSP-NEG corpus (Jiménez-Zafra et al., 2018c) will be used to train and test the systems. The quality of the dataset was measured in terms of Kappa coefficient being of 0.97 for
negation cues, 0.94 for scopes, 0.95 for events, 0.95 for the type of negation structure and 0.99 for the type of change in the polarity. A detailed discussion of the main sources of disagreements can be found in (Jiménez-Zafra et al., 2016).


-------------
CORPUS 
-------------

The SFU ReviewSP-NEG corpus (Jiménez-Zafra et al., 2018c) has been splitted into development, training and test sets. It is an extension of the Spanish part of the SFU Review corpus (Taboada et al., 2006) and it could be considered the counterpart of the SFU Review Corpus with negation and speculation annotations (Konstantinova et al., 2012). The Spanish SFU Review corpus consists of 400 reviews extracted from the website Ciao.es that belong to 8 different domains: cars, hotels, washing machines, books, cell phones, music, computers, and movies. For each domain there are 50 positive and 50 negative reviews, defined as positive or negative based on the number of stars given by the reviewer (1-2=negative; 4-5=positive; 3-star review were not included). Later, it was extended to the SFU ReviewSP-NEG corpus in which each review was automatically annotated at the token level with pos-tags and lemmas, and manually annotated at the sentence level with negation cues and their corresponding scopes and events. Moreover, it is the first corpus in which it was annotated how negation affects the words that are within its scope, that is, whether there is a change in the polarity or an increment or reduction of its value.

Training:  264 reviews, 33 per domain (cars, hotels, washing machines, books, cell phones, music, computers, and movies).

Development:  56 reviews, 7 per domain (cars, hotels, washing machines, books, cell phones, music, computers, and movies).

The corpus was automatically tokenized, PoS tagged and lemmatized using Freeling and then, it was manually annotated with negation and polarity information applying the procedure described in (Jiménez-Zafra et al., 2018c).


-------------
INPUT FORMAT
-------------

The data are provided in XML format and follow the annotation scheme described below. 

<review polarity="positive/negative">
	<!-- Sentence with negation -->
	<sentence complex="yes/no">
	        <neg_structure polarity="positive/negative/neutral" change="yes/no" polarity_modifier="increment/reduction" value="neg/noneg/comp/contrast">
			<scope>
	                        <negexp discid="1n/1c">
				</negexp>
				<event>
				</event>
			</scope>
		</neg_structure>
	</sentence>
	<!-- Sentence without negation -->
	<sentence>
	</sentence>
</review>
		

The label <review polarity> describes the polarity of the whole review, which can be positive or negative, according to the value assigned to it in the SFU Review Corpus. The label <sentence> can correspond to a complete sentence, a clause, a phrase or a fragment of a sentence with a self-contained meaning, in which a negative structure can occur. In SFU Review
SP-NEG, we only annotate the structures that contain at least one negation marker or negation cue. We assign the value "yes" to the <sentence complex> attribute when the <sentence> contains more than one negative structure (<neg_structure>) (1) and we assign the value "no" to the <sentence complex> attribute when the <sentence> contains only one negative structure (2).

	1.  <sentence complex="no"> El anterior coche se paró a la media hora de comprarlo <neg_structure> porque no le habían quitado el precinto de seguridad </neg_structure> </sentence>
	‘Our previous car stopped half an hour after we bought it because they had not removed the security seal.’

	2.  <sentence complex="yes"> <neg_structure> para que no les entre polvo </neg_structure> <neg_structure> para que no se oxiden </neg_structure></sentence>
	‘so that dust does not get in or so that they do not rust.’


Complex structures (<sentence complex="yes">) can be embedded or non-embedded. Embedded structures (3) and (4) are those in which one negative structure is part of another negative structure in the same <sentence> node. Non-embedded structures are those in which two or more negative structures appear independently in the same <sentence> node (2).

	3.  <sentence complex="yes"> <neg_structure> no quería pasarme un día entero en el aeropuerto <neg_structure> sin poder descansar </neg_structure> </neg_structure> </sentence>
	‘I did not want to spend the whole day at the airport without resting.’

	4.  <sentence complex="yes"> <neg_structure> no tenía culpa de <neg_structure> no tenerlo </neg_structure> <neg_structure> </sentence>
	‘I was not to blame for not having it.’

The label <neg_structure> is assigned to a syntactic structure -corresponding either to a sentence, a clause or a phrase-, which contains a negation marker or a negation cue. This label has four attributes associated with it:

– <polarity>: indicates the positive (5), negative (6) or neutral (7) orientation of the negative structure.
	5.  <neg_structure polarity="positive"> No vas a tener problemas </neg_structure>
	‘You will have no trouble.’

	6.  <neg_structure polarity="negative"> Segundas partes nunca fueron buenas </neg_structure>
	‘Sequels are never any good.’

	7.  <neg_structure polarity="neutral"> El realismo de Flaubert no busca la precisión histórica </neg_structure>
	‘Flaubert’s realism does not aspire to historical accuracy.’

– <change>: indicates whether the polarity (8) or the meaning (9) of the negative structure is modified or not by the negation.
	8.  <neg_structure polarity="positive" change="yes"> La calidad del sonido no es mala </neg_structure>
	‘The sound quality is not bad.’

	9.  <neg_structure polarity="negative" change="yes"> Ni siquiera tengo carnet </neg_structure>
	‘I do not even have a card.’

– <polarity_modifier>: indicates whether the negative structure contains an element that modifies or nuances its polarity (e.g.: ‘chico  bueno’/’good boy’ versus ‘chico  no  muy  bueno’/’not very good boy’). This attribute has two possible values: "increment" to indicate an increment in the polarity value (10) and "reduction" to indicate a diminishing of the polarity value (11).

	10.  <neg_structure polarity="positive" polarity_modifier="increment"> No me arrepiento para nada <neg_structure>
	‘I do not regret (it) at all.’

	11.  <neg_structure polarity="negative" polarity_modifier="reduction"> No lo he utilizado mucho </neg_structure>
	‘I have not used it much.’

– <value>: indicates the meaning expressed by the negative structure. It has four possible values:
	–  "neg" indicating negation (12);
	–  "contrast" indicating contrast or opposition between terms (13);
	–  "comp" expressing a comparison or inequality between terms (14);
	–  "noneg" indicating structures that contain a negative particle but which do not negate (15).

	12. <neg_structure value="neg"> El aire acondicionado ni enfría ni calienta <neg_structure>
	‘The air conditioning doesn’t heat or cool.’


	13. <neg_structure value="contrast"> No vinieron 2 soldados, sino 6 <neg_structure>
	‘6 soldiers came, not 2.’

	14. <neg_structure value="comp"> El ambiente de este lugar es agradable pero no tanto como el del otro <neg_structure>
	‘The atmosphere in this place is pleasant but not as much as in the other one.’

	15. <neg_structure value="noneg"> El coche lo compré para viajar, ¿no? <neg_structure>
	‘I bought this car for travelling, didn’t I?’

The label <scope> delimits the part of the negative structure that is within the scope of the negation. It includes both the negative marker or cue <negexp> and the event <event>.
– <negexp> includes the word(s) that expresses negation. Negation can be expressed by one or more than one negative element. In the latter case, the elements can be continuous or discontinuous and the second negative element usually nuances the first one (16). When they are discontinuous, we identify the negative elements by means of two <negexp> labels, each of them with the attribute <discid> (discontinuity id). The value of <discid> is represented both numerically (1 in the example below), which indicates the numerical order of the discontinuous negative elements in that negative structure, and as a letter "n" and "c", where "n" and "c" indicate the first (nucleus or core) and second element of the negation respectively.

	16.  El coche <negexp discid="1n"> no </neg_exp> <event> frena </event> <negexp discid="1c"> en absoluto </negexp>
	‘The car does not brake at all.’

In the case of coordinated negations, the <discid> label identifies the different coordinated negative elements (17).

	17. Permiten el paso <negexp discid="1n"> sin </neg_exp> <event> grandes contorsiones <negexp discid="1c"> ni </negexp> aspavientos </event>
	‘They allow one to pass without major contortions and without fuss.’

The label <discid> is also used in discontinuous negative structures expressing contrast (18) or comparison (19).

	18.  <neg_structure value="contrast" polarity="neutral" change="no"> La segunda parte del libro, <negexp discid="1c"> lejos de </negexp> mantener mi entusiasmo <negexp discid="1n"> más bien </negexp> lo sepultó </neg_structure>
	‘The second part of the book, far from maintaining my enthusiasm, killed it off (instead).’

	19.  <neg_structure value="comp" polarity="positive" polarity_modifier="reduction"> Su exterior <negexp discid="1n"> no </neg_exp> me gusta <negexp discid="1c"> tanto como </negexp> el
	de otras marcas </neg_structure>
	‘I do not like the outside of it as much as that of other brands.’

– <event> indicates the word(s) directly negated/affected by the negative marker. It is usually a part of the scope, though it can also match the scope. The <event> can be a verb, a noun (20) or an adjective (21). Verbs can be a simple (22) or complex verbal form, such as a passive verbal form (23), a periphrastic verbal form (24) or a light verb (25).

	20.  <negexp> Cero </negexp> <event> fiabilidad </event>
	‘Zero reliability.’

	21.  <negexp> Nada </negexp> <event> bueno </event>
	‘Not good at all.’

	22.  <negexp> No </negexp> <event> hablo </event> de accesorios
	‘I am not speaking about accesories.’

	23.  Mis peticiones <negexp> no </negexp> <event> fueron atendidas </event>
	‘My requests were not addressed.’

	24.  <negexp> No </negexp> <event> deseo regresar </event> a ese hotel
	‘I do not want to go back to that hotel.’

	25.  El modelo <negexp> no </negexp> <event> da problemas </event>
	‘The model does not create problems.’

In the case of pronominal verbs (26) or verbs with a pronoun in the passive voice (27), the pronouns are also included inside the <event>, because they are part of the verbal form.

	26.  por lo que <negexp> no </negexp> <event> te mareas </event>
	‘so you do not feel sick.’

	27.  aunque ya <negexp> no </negexp> <event> se fabrica </event>
	‘although it is no longer being made.’

The complements of copulas (28) and predicative complements (29) are also included inside the <event>, because the semantic content is in the complement (they are basically adjectives).

	28.  <negexp> No </negexp> <event> es pesado </event>
	‘It is not heavy.’

	29.  <negexp> No </negexp> <event> resulta agradable </event>
	‘It is not pleasant.’

Finally, the elliptical <event> is identified with the empty symbol set and manually tagged with the attribute <elliptic>. In example (30), the antecedent of the elliptical event is "tiene un coche de segunda mano machacao" (‘you have a beaten-up second-hand car’).

	30.  Os preguntaréis: que tiene un coche de segunda mano machacao? <neg_structure> Pues <negexp> no </negexp> <event> Ø <elliptic=‘‘yes’’> </event> señor </neg_structure>
	‘You are asking yourselves: What is so special about this beaten-up second-hand car? Well, no sir, I am not.’


-------------
OUTPUT FORMAT
-------------

Systems have to output one file with the polarity of each filename using the following format:

filename\tdomain\tpolarity

Example:

	no_2_7	libros	negative
	no_1_9	libros	negative
	yes_4_3	libros	positive
	no_1_12	libros	negative
	no_2_24	libros	negative
	yes_4_20	libros	positive
	yes_5_15	libros	positive
	yes_4_1	moviles	positive
	yes_4_17	moviles	positive
	no_2_3	moviles	negative
	no_2_16	moviles	negative
	yes_4_16	moviles	positive
	no_2_17	moviles	negative
	no_2_13	moviles	negative

------------
EVALUATION
------------

The following evaluation measures are used to evaluate the systems:

	- Precision
	- Recall
	- F1-score
	- Accuracy

The evaluation script measures precision, recall and F1-score per class and averages them using macro-average method. Moreover, it also provides accuracy.

 precision = tp / (tp + fp)
 recall = tp / (tp + fn)
 F1 = (2 * $precision * $recall) / ($precision + $recall)

 Accuracy = (tp + tn) / (tp + tn + fp + fn)
 

For evaluating, use the file scorer_task3.py. You can execute this python command with -h to obtain help:

>python3 scorer_task3.py -h

Usage: python3 scorer_task3.py [OPTIONS] -g <gold standard> -s <system output>

optional arguments:
  -h, --help            show this help message and exit

required named arguments:
  -g GOLD, --gold GOLD  Gold standard
  -s SYSTEM, --system SYSTEM System output


-------------
REFERENCES
-------------

Farkas, R., Vincze, V., Móra, G., Csirik, J., & Szarvas, G. (2010, July). The CoNLL-2010 shared task: learning to detect hedges and their scope in natural language text. In Proceedings of the Fourteenth Conference on Computational Natural Language Learning---Shared Task (pp. 1-12). Association for Computational Linguistics.

Jiménez-Zafra, S. M., Martín-Valdivia, M. T., Ureña-Lopez, L. A., Marti, T., & Taulé, M. (2016). Problematic cases in the annotation of negation in Spanish. In Proceedings of the Workshop on Extra-Propositional Aspects of Meaning in Computational Linguistics (ExProM) (pp. 42-48).

Jiménez-Zafra, S. M., Díaz, N. P. C., Morante, R., & Martın-Valdivia, M. T. (2018a). Tarea 1 del Taller NEGES 2018: Guías de Anotación. In Proceedings of NEGES 2018: Workshop on Negation in Spanish (Vol. 2174, pp. 15-21).

Jiménez-Zafra, S. M., Cruz-Díaz, N. P., Morante, R., & Martín-Valdivia, M. T. (2018b). Tarea 2 del Taller NEGES 2018: Detección de Claves de Negación. In Proceedings of NEGES 2018: Workshop on Negation in Spanish (Vol. 2174, pp. 35-41).

Jiménez-Zafra, S. M., Taulé, M., Martín-Valdivia, M. T., Ureña-López, L. A., & Martí, M. A. (2018c). SFU Review SP-NEG: a Spanish corpus annotated with negation for sentiment analysis. A typology of negation patterns. Language Resources and Evaluation, 52(2), 533-569. First published online May 22, 2017. https://doi.org/10.1007/s10579-017-9391-x

Konstantinova, N., & De Sousa, S. C. (2011). Annotating negation and speculation: the case of the review domain. In Proceedings of the Second Student Research Workshop associated with RANLP 2011 (pp. 139-144).

Morante, R., & Blanco, E. (2012, June). * SEM 2012 shared task: Resolving the scope and focus of negation. In Proceedings of the First Joint Conference on Lexical and Computational Semantics-Volume 1: Proceedings of the main conference and the shared task, and Volume 2: Proceedings of the Sixth International Workshop on Semantic Evaluation (pp. 265-274). Association for Computational Linguistics.

Morante, R., & Sporleder, C. (2010). Proceedings of the Workshop on Negation and Speculation in Natural Language Processing. In Proceedings of the Workshop on Negation and Speculation in Natural Language Processing.

Taboada, M., Anthony, C., & Voll, K. (2006, May). Methods for creating semantic orientation dictionaries. In Conference on Language Resources and Evaluation (LREC) (pp. 427-432).


