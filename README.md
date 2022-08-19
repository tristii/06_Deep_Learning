# 06_Deep_Learning

Binder Badge zum starten des Jupyter Notebooks (Keras-Projekt_Musterloesung.ipynb) via myBinder: 
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/tristii/06_Deep_Learning/main?labpath=Keras_Projekt_Musterloesung.ipynb)


In der Umgebung von mybinder können Sie das Übungsbeispiel ausführen lassen, eigene Lösungen codieren (bestehenden Code modifizieren), um das Übungsbeispiel mit dem erwartenden Ergebnis zu reproduzieren.

## Dokumentation zur Übungsaufgabe: 
Aufgebaut ist die Übungsaufgabe in jeweils in Beschreibungsabschnitt was explizit gemacht werden soll. Die jeweilige darunterliegende Zelle stellt den Code Bereich dar, in der dieser Aufgabenabschnitt codiert wird und ggf. den dazugehörigen Output nach der Ausführung (SHIFT + Enter) der jeweiligen Codezelle darstellt.

! Um das zu erwartendee Ergebnis (Output) der Lösung im Übungsbeispiel nicht zu verlieren bzw. zu überschreiben, können Sie nach der Beschreibung der jeweiligen Teil-Aufgabe eine Zwischenzeile(neue Zeile) einfügen, in der Sie Ihre Lösung codieren um danach Ihren Code mit dem darunter liegenden Mustercode + dessen Output vergleichen/überprüfen zu können. 

**Vorgehensweise im Übungsbeispiel:**

Fallbeispiel: Der Datensatz

Wir werden einen Teil des LendingClub-Datensatzes von Kaggle verwenden: https://www.kaggle.com/wordsforthewise/lending-club
LendingClub ist ein US-basierter Peer-To-Peer-Kreditgeber mit Sitz in San Francisco, California.[3] Es war der erste Peer-to-Peer-Kreditgeber, der seine Angebote als Wertpapiere bei der Securities and Exchange Commission (SEC) registrierte, und der Kredithandel auf einem Sekundärmarkt anbietet. LendingClub ist der weltgrößte Peer-to-Peer-Kreditgeber. 

Unser Ziel: Können wir mit historischen Daten über ausgegebene Kredite und der Informationen, ob der Kreditnehmer den Kredit zurückgezahlt hat, vorhersagen, ob ein neuer Kreditnehmer seinen Kredit zurückzahlen wird? Dadurch könnten wir zukünftige, potenzielle Kreditnehmer bewerten nach der Wahrscheinlichkeit, dass sie ihren Kredit begleichen. Beachte die Klassifikationsmetriken bei der Bewertung der Leistung deines Modells!
Die Spalte "loan_status" enthält unser Label.

**1. Libraries**
* notwendige Libraries müssen importiert werden (pandas, numpy, matplotlib.pyplot, seaborn) für die Datenverarbeitung und Datenvisualisierung

**2. Rohdaten**
* Rohdaten (Lending_club_info.csv; Lending_club_loan_two.csv) einlesen und DataFrame erstellen
* Deskriptive Statistiken anzeigen und analysieren mit der head(), info() und describe() Funktion
* Erstelle eine neue Spalte namens "text length", welche die Anzahl von Zeichen der "text" Spalte beinhaltet.

**3. Explorative Datenanalyse**

mittels Plots aus der Seaborn und Pandas Library können die Daten visualisiert werden, um sie besser analysieren zu können. 
* Countplot von ("loan_status") erstellen
* Histogramm der Spalte ("loan_amnt) erstellen
* Berechne die Korrelation aller kontinuierlichen, numerischen Variablen mit der Methode corr()
* Heatmap der Korrelation ausgeben 
*  Das "installment"-Feature sollte eine beinahe perfekte Korrelation aufweisen. Zeige die Beschreibung der 2 Features "installment" und "loan_amnt" mit feat_info() an
*  Erstelle ein Scatter Plot von "installment" und "loan_amnt"
*  Erzeuge ein Boxplot zur Darstellung der Beziehung zwischen "loan_status" und loan_amnt"
*  Berechne die zusammengefassten Statistiken für "loan_amnt", gruppiert nach loan_status
*  Zeige die einzigartigen Klassen und Unterklassen der Spalten "grade" und "sub_grade" auf
*  Stelle in einem Countplot die Anzahl pro Klassen dar. Hue (Aufteilung) soll auf "loan_status" gesetzt werden
*  Stelle die Anzahl pro Unterklasse in einem Countplot dar
*  Denselben Plot erstellen aber Hue auf "loan_status" setzen 
*  Countplot der "Subgrade" Spalte, gefiltert nach Grade F und G und unterteilt nach "loan_status"
*  Neue Spalte "loan_repaid erstellen. Erhält binäre Wert 0 wenn der Staus "Charged Off" ist und 1 wenn der Status des Kredits "Fully Paid" ist.
*  Barplot zur Korrelation der numerischen Features mit der Spalte "loan_repaid" erstellen 
 
 
**4. Data Preprocessing**

Umgang mit fehlenden Daten.
* Fehlende Werte des DataFrame anzeigen lassen
* Konvertierung in Prozent des gesamten DataFrame
* Anzeigen der einzigartigen Arbeitsplatzbezeichnungen "emp_title" 
* Spalte "emp_title" entfernen 
* Plot erstellen mit der Anzahl der Spalte des Features emp_length. Sortiere die Reihenfolge der Werte
* Countplot mit Fully Paid gegen Charged Off mit Hue= "loan_status" erstellen
* Untersuchung auf einen Zusammenhang zwischen Anstellungsdauer und Rückzahlungen
* Barplot des Zusammenhangs erstellen 
* Spalte "emp_length" verwerfen
* Betrachte mit head() Spalten "purpose" und "title"
* Verwerfe Spalte "title"
* Beschreibung des Features "mort_acc" anzeigen 
* Erzeuge ein value_counts der Spalte mort_acc
* Korrelation der Features mit "mort_acc" anzeigen
* Gruppierung der Spalte "total_acc" und deren Durchschnittswerte
* Ersetze die fehlenden Werte von mort_acc basierend auf deren total_acc-Werte. Wenn mort_acc fehlt, wird der fehlende Wert mit dem Durchschnitt entsprechend des total_acc-Wertes aus der oben erzeugten Series ersetzt
* Entfernen der Zeilen mit fehlenden Werten mit dropna() Methode


**5. Kategorische Variablen und Dummyvariablen**
* Nicht numerische Spalten anzeigen 
* Konvertiere das Feature "term" in den numerischen Datentyp Integer
* Verwerfe Spalte "grade"
* Konvertiere "subgrade", "verification_status", "application_type", "initial_list" und "purpose" in Dummyvariablen. Orginal Spalten werden verworfen 
* Betrachte die value_counts der Spalte "home_ownership"
* Konvertiere diese Dummyvariablen, aber ersetze NONE und ANY mit OTHER, so dass es nur 4 Kategorien gibt
* Erzeuge eine Spalte 'zip_code', dass die Postleitzahl aus der Spalte address extrahiert
* Verwerfe Spalte "issue_d"
* Extrahiere das Jahr von "earliest_cr_line" mit der Funktion apply() und konvertiere es in ein numerisches Feature. Schreibe diese Daten in eine Featurespalte 'earliest_cr_year' und verwirf dann das Feature earliest_cr_line.


**6. Aufteilung in Trainings und Testdaten**
* Importiere train_test_split von sklearn
* Verwerfe die Spalte "loan_status"
* Setze die Variablen X und y zu den .values der Features und Label
* Teile auf in Trainings- und Testdaten mit test_size=0.2 und random_state=101

**7. Daten normalisieren**
* import MinMaxScaler von sklearn.preprocessing 
* Normalisiere mit dem MinMaxScaler die Featuredaten in X_train und X_test 

**8. Modell erzeugen**
* importiere die nötigen Libraries von Tensorflow
* Erzeuge ein sequenzielles Modell, das mit den Daten trainiert wird
* Folgendes Modell wird hier verwendet: Input Layer, d.h. die Anzahl der Anfangsknoten sind hier 78.
2x Hidden Layer, d.h. die Layer zwischen Input und Output. 1. HL hat 39 Knoten, 2. HL hat 19 Knoten.
Output Layer mit einem Ergebnis.
* Weitere Infos zum Modell: "Dropouts" geben die Gewichtung/Wichtigkeit der Kanten an.
"Activation" sind die Funktionen, die verwendet werden. Hier ReLU und Sigmoid Funktionen
* Fitte das Model, passe hierzu für mindestens 25 Epochen das Modell an die Trainingsdaten an. Füge die Validierungsdaten hinzu für eine spätere Darstellung

**9. Evaluation der Leistung des Modells** 
* Plotte den loss der Validierung gegen den loss des Trainings
* import classification_report,confusion_matrix von sklearn.metrics
* Erzeuge Vorhersagen mittels des Datensatzes X_test 
* Klassifizierungsreport für das Modell erstellen, um die Perfomance in Bezug auf Precision und Recall Werte zu erhalten
* Confusion Matrix erstellen, um die Leistung des Algorithmus visualisieren zu können
