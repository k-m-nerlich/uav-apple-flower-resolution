# Annotationsprotokoll — Apfelblüten in Drohnenbildern

Für die zweite annotierende Person. Bitte **vollständig lesen, bevor du die
erste Kachel bearbeitest.** Die Regeln sind nicht beliebig: Gemessen wird, wie
gut zwei Personen unabhängig voneinander zum selben Ergebnis kommen. Nur wenn
beide nach denselben Regeln arbeiten, sagt das Ergebnis etwas über die Aufgabe
statt über Missverständnisse.

## Was zu tun ist

Du bekommst 36 Bildausschnitte aus einer Apfelanlage, aufgenommen mit einer
Drohne aus etwa 5 m Höhe. Jeder Ausschnitt zeigt rund 1,7 × 1,7 m Boden mit
einem Stück Baumreihe.

**Setze einen Punkt auf jede Apfelblüte, die du erkennst.**

## Die Regeln

1. **Ein Punkt pro Einzelblüte**, gesetzt in die Blütenmitte. Nicht ein Punkt
   pro Büschel. Wenn du fünf Blüten dicht beieinander siehst, setzt du fünf
   Punkte.
2. **Am Bildrand angeschnitten**: zählen, wenn mehr als die Hälfte der Blüte
   sichtbar ist.
3. **Im Schatten**: zählen, wenn du sie als Blüte erkennst.
4. **Knospen und verblühte Blüten** zählen nicht. Nur offene weiße Blüten.
5. **Zweifelsfall**: Label `unsicher` statt raten. Diese Punkte werden separat
   ausgewertet — ihre Anzahl ist ein Ergebnis, kein Ausschuss. Lieber einmal zu
   oft `unsicher` als eine Zufallsentscheidung.
6. **Leere Kacheln gibt es.** Wenn nichts zu sehen ist, speicherst du ohne
   einen einzigen Punkt. Das ist richtig so.

## Wichtig für die Auswertung

- **Nicht länger suchen, als du beim ersten Blick brauchst.** Setz dir ein
  Zeitlimit, etwa 2 bis 3 Minuten pro Kachel, und halte es durch. Gemessen
  wird, was bei normaler Betrachtung erkennbar ist, nicht was mit maximaler
  Anstrengung noch herauszuholen wäre.
- **Keine fremden Annotationen ansehen.** Weder vorher noch zwischendurch.
  Sobald du weißt, wo jemand anders Punkte gesetzt hat, ist der Vergleich wertlos.
- **Nicht zurückgehen und frühere Kacheln überarbeiten**, nachdem du dazugelernt
  hast. Arbeite einmal durch.
- **Reihenfolge einhalten**, wie sie dir angezeigt wird.

Zoomen ist erlaubt und sinnvoll — die Blüten sind klein. Zoomen ist kein
"länger suchen", sondern normales Betrachten.

## Technischer Ablauf

1. Label Studio installieren:
   `pip install label-studio`
2. Starten:
   `label-studio start --data-dir ./ls_data`
   Browser öffnet auf `http://localhost:8080`, Konto anlegen (rein lokal).
3. **Create Project**, Name z. B. `Bluete Annotator 2`.
4. Reiter **Labeling Setup** → ganz unten **Custom template** → Inhalt von
   `labelconfig.xml` einfügen, speichern.
5. Reiter **Data Import** → **Upload Files** → alle JPG aus dem Ordner
   `kacheln/` hochladen.
6. Annotieren. Taste `1` wählt `Bluetencluster` (das ist die Einzelblüte, der
   Name ist historisch), Taste `2` wählt `unsicher`. Klick setzt einen Punkt,
   `Strg+Z` macht rückgängig, `Strg+Enter` speichert und geht weiter.
7. Wenn alle Kacheln bearbeitet sind: **Export** → Format **JSON** →
   Datei zurückschicken.

## Zeitbedarf

Erfahrungswert aus der ersten Runde: Median 1,5 Minuten pro Kachel. Für 36
Kacheln also gut eine Stunde, plus Einrichtung.

## Noch etwas

Notier bitte kurz, ob du vorher schon einmal mit Obstbau oder Luftbildern zu
tun hattest. Das gehört als Angabe ins Paper — nicht als Bewertung, sondern
weil es für die Einordnung des Ergebnisses zählt.
