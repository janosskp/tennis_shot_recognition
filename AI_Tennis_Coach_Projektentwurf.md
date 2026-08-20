# AI Tennis Coach – Projektentwurf
### Real-Time Tennis Shot Recognition & Performance Analysis
*Erstellt als Entscheidungsgrundlage für ein 4-Wochen-Hochschulprojekt (1 Hauptperson à 20h/Woche, 2 unterstützende Personen)*

---

## Executive Summary

Ziel ist ein **Pose-basiertes Shot-Recognition-System**, das aus Tennisvideos die Schlagarten **Forehand, Backhand, Serve** erkennt und die Ergebnisse in einem interaktiven Video-Dashboard darstellt. Statt eines komplexen End-to-End-Video-Deep-Learning-Modells wird eine **zweistufige, leichtgewichtige Pipeline** gewählt:

1. **Vortrainierte Pose-Estimation** (MediaPipe Pose oder YOLOv8-Pose, kein eigenes Training nötig) extrahiert Körper-Keypoints pro Frame.
2. Ein **selbst trainierter, kleiner Klassifikator** (MLP bzw. leichtes 1D-CNN über ein Sliding Window von Keypoint-Sequenzen) ordnet daraus die Schlagart zu.

Diese Architektur wurde gewählt, weil sie gegenüber End-to-End-Video-Modellen (z. B. 3D-CNN, Video-Transformer) und gegenüber reiner Objekterkennung (YOLOv8 auf Rohbildern) den **höchsten erwarteten Bewertungswert pro investierter Stunde** liefert: sie benötigt kaum Bounding-Box-Annotation, läuft auf einer CPU in Echtzeit-Nähe, ist gut erklärbar (Bewegungsmuster sichtbar machbar) und ist mit realistischem Zeitbudget tatsächlich trainierbar.

„Shot Success Rate" wird **nicht als gemessenes MVP-Feature**, sondern als ehrlich begründetes Stretch Goal behandelt, da eine zuverlässige Erfolgsdefinition aus reinen RGB-Videos ohne Ballverfolgung/Court-Tracking nicht robust ableitbar ist.

Die Demo läuft **lokal, offline, video-basiert** (kein Live-Webcam-Zwang), mit echten Interaktionselementen (Video-/Shot-Auswahl, Timeline, Confidence-Threshold, Statistik-Dashboard) und einer vorbereiteten Fallback-Version für den Fall technischer Probleme während der Präsentation.

---

## 1. Problem & Motivation

Amateur- und Vereinsspieler erhalten selten objektives, quantifiziertes Feedback zu ihrem Spiel. Trainer sind teuer, subjektiv und nicht immer verfügbar. Ein System, das automatisiert erkennt, **welche Schlagart** wie oft gespielt wurde, ist ein realistischer, nützlicher erster Baustein eines datengetriebenen Trainingsassistenten – ohne zu behaupten, ein KI-System könne bereits "coachen".

Der Use Case ist real-world-relevant, weil er:
- auf existierendem, frei verfügbarem Videomaterial funktioniert,
- keine spezielle Sensorik (Wearables, Multi-Kamera-Setups) voraussetzt,
- als Grundlage für spätere Erweiterungen (Trefferanalyse, Lauf-/Positionsanalyse) dient.

**Wichtige Einschränkung, die wir offen kommunizieren:** Das System erkennt Schlagarten und stellt sie interaktiv dar – es bewertet nicht die Schlagqualität und ist kein Ersatz für menschliches Coaching.

---

## 2. Project Concept

**Titel:** AI Tennis Coach – Shot Recognition & Performance Overview

**Kernfunktion:**
Video rein → Pose-Extraktion pro Frame → Sliding-Window-Klassifikation (Forehand/Backhand/Serve/kein Schlag) → zeitliches Glätten (Debouncing, um Flackern zu vermeiden) → interaktives Overlay + Statistik-Dashboard.

**Abgrenzung:** Kein Live-Webcam-Zwang, kein Ballverfolgung/Trefferanalyse im MVP/Target, keine Gesichtserkennung, keine Spieleridentifikation.

---

## 3. Grading Strategy – Zahlungspunkte auf die 6 Kategorien

| Kategorie | Gewicht | Wie das Projekt einzahlt |
|---|---|---|
| Concept & Creativity / Real-World Relevance | 15% | Konkreter, nachvollziehbarer Trainings-Use-Case; ehrliche Abgrenzung statt Buzzwording |
| Technical Implementation & Interactivity | 25% | Video-Pipeline mit nahezu Echtzeit-Verarbeitung auf CPU, echte Interaktionselemente (s. Kap. 16), stabile Demo |
| AI Model & Technical Grounding | 20% | Bewusste, begründete Architekturwahl, eigener Trainingslauf, Baseline-Vergleich, Accuracy/F1/Latenz-Messung |
| Documentation & Reproducibility | 15% | Modulares Repo, README mit <1h-Setup, getrennte Daten-/Modellablage |
| Ethical & Societal Reflection | 10% | Konkrete Risiken (Videodaten von Personen, Bias, Missbrauch) + Gegenmaßnahmen |
| Presentation & Communication | 15% | Klares 6-Punkte-Demo-Skript, Fallback-Strategie, präzises Wording |

---

## 4. Technical Architecture

**Grundproblem (Kap. 6 des Auftrags):** Ein Schlag ist eine zeitliche Bewegung, kein statisches Objekt. Fünf Ansätze im Vergleich:

| Ansatz | Vorteil | Nachteil |
|---|---|---|
| Einzelbildklassifikation (rohes RGB-Frame → CNN) | Einfach, wenig Daten nötig | Ignoriert Bewegung, verwechselt leicht Vor-/Nachschwung, geringe Erklärbarkeit |
| Object Detection (YOLOv8 auf Ball/Spieler/Schläger) | Bekannt aus eigener Erfahrung, gute Tools | Erkennt "was ist im Bild", nicht "welche Bewegung"; braucht Bounding-Box-Annotation |
| Pose-/Keypoint-basierte Klassifikation (Einzelframe) | Reduziert Rohbild auf robuste Körperhaltung, kaum Annotation nötig (nur Klassenlabel), gut erklärbar | Einzelne Pose kann mehrdeutig sein (z. B. Ausholphase Forehand vs. Backhand) |
| Temporale Action Recognition (3D-CNN/Video-Transformer auf Rohvideo) | Höchste theoretische Genauigkeit, "state of the art" | Braucht viele Trainingsdaten, GPU-intensiv, in 4 Wochen ohne GPU nicht seriös trainierbar |
| **Kombination: Pose-Estimation (vortrainiert) + leichter temporaler Classifier auf Keypoint-Sequenzen** | Kombiniert Robustheit der Pose-Reduktion mit Bewegungsinformation; sehr wenig Trainingsdaten/-rechenleistung nötig; CPU-fähig; erklärbar | Erkennungsqualität hängt an Qualität der Pose-Schätzung (z. B. bei Verdeckung) |

Der letzte Ansatz wird empfohlen (siehe Kap. 6).

---

## 5. Architecture Comparison (Pflichtvergleich, 2 realistische Kandidaten)

| Kriterium | A: YOLOv8 (Object Detection auf Rohframes, Spieler/Schläger-Boxen) + heuristische/CNN-Klassifikation | B: Pose Estimation (MediaPipe/YOLOv8-Pose, vortrainiert) + leichter Sequenz-Classifier (MLP/1D-CNN) |
|---|---|---|
| Technische Funktionsweise | Detektiert Objekte, Schlagart müsste aus Boxen/Heuristik oder nachgeschaltetem CNN abgeleitet werden | Extrahiert 17–33 Körper-Keypoints/Frame, Classifier lernt Bewegungsmuster über Zeitfenster |
| Benötigte Daten | Viele annotierte Frames mit Bounding Boxes pro Klasse | Nur Klassenlabel pro Clip/Zeitfenster (kein Bounding Box nötig, da Pose-Modell vortrainiert) |
| Annotation-Aufwand | Hoch (Bounding Boxes pro Frame) | Niedrig (Clip-Label reicht) |
| Trainingsaufwand | Mittel–hoch (Fine-Tuning Detection-Head) | Niedrig (kleiner Classifier, Minuten bis wenige Stunden CPU/Colab) |
| Inferenzgeschwindigkeit (CPU, Laptop) | Mittel, YOLOv8 auf CPU spürbar langsamer bei Videoauflösung | Gut: Pose-Modelle sind für Echtzeit auf CPU optimiert (MediaPipe explizit dafür gebaut) |
| Implementierungsaufwand | Mittel (Annotation-Tool nötig, Trainingspipeline) | Niedrig–mittel (Pose-Modell fertig, nur Classifier + Windowing selbst gebaut) |
| Hardwareanforderungen | Fine-Tuning idealerweise mit GPU (Cloud) | Classifier-Training auch auf CPU machbar, Cloud optional |
| Erklärbarkeit | Mittel (Boxen sichtbar, aber Schlagart-Logik unklar) | Hoch (Keypoint-Trajektorien direkt visualisierbar, z. B. Handgelenk-Bahn) |
| Robustheit | Abhängig von Beleuchtung/Perspektive der Boxen | Pose-Modelle sind auf vielfältigen Datensätzen vortrainiert, robuster gegenüber Szenenwechsel |
| Risiko | Höher: Annotation-Aufwand kann Zeitbudget sprengen | Niedriger: Kernkomponente ist bereits vortrainiert und getestet |
| Demo-Eignung | Boxen + Label wirken bekannt, aber weniger "intelligent" | Skelett-Overlay wirkt visuell überzeugend und nachvollziehbar erklärbar |
| Erwartete ML-Performance | Mittel, abhängig von Annotationsqualität | Erwartet gut bis sehr gut für 3 klar unterscheidbare Klassen bei sauberem Datensatz (Hypothese, nicht gemessen) |
| Erwarteter Bewertungsbeitrag | Solide, aber Risiko durch Annotationsaufwand senkt Interactivity/Grounding-Score | Höher: setzt Ressourcen auf Interactivity/Evaluation statt auf Annotation |

**Entscheidungsmatrix (qualitativ, 1=schlecht, 5=sehr gut):**

| Kriterium | A: YOLOv8-Detection | B: Pose+Sequenz-Classifier |
|---|---|---|
| Umsetzbarkeit in 4 Wochen | 3 | 5 |
| Annotationsaufwand (invers) | 2 | 5 |
| CPU-Demofähigkeit | 3 | 5 |
| Erklärbarkeit | 3 | 5 |
| Risiko (invers) | 3 | 4 |
| **Summe** | **14** | **24** |

---

## 6. Recommended Architecture

**Empfehlung: Ansatz B – Pose Estimation + leichter Sequenz-Classifier.**

Begründung nach *Expected Grade Value / Effort / Risk / Technical Quality*:
- **Expected Grade Value:** hoch – zahlt direkt auf Interactivity (Echtzeit-fähig), Grounding (klar erklärbares Modell) und Reproducibility (kein riesiges annotiertes Datenset nötig) ein.
- **Effort:** niedrig–mittel – Pose-Modell ist fertig, nur der kleine Classifier muss selbst trainiert werden.
- **Risk:** niedrig – Kernrisiko (Pose-Erkennung) ist bereits durch etablierte Bibliotheken gelöst; eigenes Risiko beschränkt sich auf den einfachen Classifier.
- **Technical Quality:** ausreichend hoch für den Anspruch des Kurses, ohne unrealistisches Overengineering.

Damit wird explizit gegen einen komplexeren End-to-End-Video-Transformer/3D-CNN-Ansatz entschieden – dieser wäre auf dem verfügbaren Datensatz und der verfügbaren Hardware in 4 Wochen nicht seriös trainierbar (Priorität 1: stabiles Projekt schlägt technische Komplexität, Kap. 30/31 des Auftrags).

---

## 7. MVP – muss funktionieren

**Input:** Einzelnes Frame (bzw. kurzes festes Zeitfenster von z. B. 15–20 Frames um einen Zeitpunkt) → Pose-Keypoints → Klassifikator → Klasse (Forehand/Backhand/Serve/kein Schlag).

- Modell ist fine-getuned/trainiert (kleiner Classifier auf Keypoint-Features), nicht nur eine Heuristik.
- Messbare Metriken: Accuracy, Precision/Recall/F1 pro Klasse, Confusion Matrix auf getrenntem Testset.
- MVP funktioniert unabhängig davon, ob die volle Video-Pipeline (Kap. 15) stabil läuft – es reicht ein Skript, das ein Testset klassifiziert und Metriken ausgibt.

## 8. Target – soll erreicht werden

**Input:** Ganzes Tennisvideo → Sliding-Window über Pose-Sequenzen → Shot Recognition pro Zeitfenster → Overlay + interaktives Dashboard, möglichst nahe Echtzeit (CPU-Laptop).

Beispiel-Dashboard (Ausbaustufe des Vorschlags aus dem Auftrag):

```
AI TENNIS COACH — Live Analysis

Current Shot:      FOREHAND        Confidence: 91%
Skeleton Overlay:   [x]  Threshold: [====------] 0.60

Detected Shots (this video)
  Forehand   ██████████████  12
  Backhand   █████████        8
  Serve      █████            5

Timeline:  |--F--|--F--|-B-|----S----|--F--|   (klickbar, springt zu Zeitpunkt)
Processing: 17–22 FPS   |   Video: 00:42 / 02:10
```

## 9. Stretch Goals

- Echte temporale Action Recognition (z. B. kleines LSTM/Temporal-CNN statt Sliding-Window-MLP) zur Verbesserung der Übergangserkennung.
- Shot Success Rate – **nur**, wenn eine ehrliche Operationalisierung gelingt (siehe Kap. 16 des Auftrags / Kap. „Shot Success" unten).
- Multi-Video-Vergleich / einfache Spieler-Statistikvergleiche.

Stretch Goals gefährden nie MVP/Target (Gate 4, Kap. 23).

---

## 10. Dataset Strategy

**Empfohlene Basis:** öffentlich verfügbare, für Forschung freigegebene Tennis-Action-Datensätze mit bereits vorhandener Klassenlabelung (z. B. Datensätze wie THETIS, die Tennisbewegungen inkl. Forehand/Backhand/Serve von mehreren Spielern in unterschiedlichen Aufnahmen enthalten), ergänzt um wenige selbst zugeschnittene Clips aus frei lizenzierten Quellen (z. B. Creative-Commons-markierte Trainingsvideos), um Robustheit zu testen.

**Zu prüfen vor Nutzung (Woche 1, Gate 1):**
- Lizenzbedingungen (Forschungs-/Bildungsnutzung vs. Weiterverbreitung im Repo)
- Tatsächliche Klassenverteilung (Forehand/Backhand/Serve i. d. R. nicht gleich häufig)
- Anzahl verschiedener Spieler und Kameraperspektiven (für Train/Test-Split entscheidend)

**Split-Strategie:** Aufteilung **auf Video-/Spieler-Ebene**, nicht auf Frame-Ebene – Frames desselben Clips dürfen nie gleichzeitig in Training und Test liegen (Data-Leakage-Vermeidung, explizit gefordert). Ziel-Split: ca. 70/15/15 (Train/Val/Test) über Videos, nicht über Frames.

**Umfang (realistisch statt maximal):** Zielgröße orientiert an "gerade genug für einen sauberen, kleinen Classifier" – z. B. einige hundert gelabelte Zeitfenster pro Klasse, nicht zehntausende. Die genaue Zahl wird nach Sichtung der verfügbaren Quellen in Woche 1 festgelegt (Gate 1).

**Variationen, die dokumentiert/geprüft werden:** Spieler, Kameraperspektive (meist Grundlinie/seitlich), Beleuchtung, Indoor/Outdoor, Spielniveau – als Grundlage für die Robustheits-KPIs (Kap. 15 unten).

---

## 11. Annotation Strategy

Kritische Vorprüfung (wie gefordert): **Nein**, für Ansatz B wird keine umfangreiche Bounding-Box-Annotation benötigt, da die Pose-Estimation vortrainiert ist. Notwendig ist nur:

- **Clip-/Zeitfenster-Label** (welche Schlagart, ggf. "kein Schlag") – wenn möglich direkt aus bereits gelabelten Datensätzen übernommen.
- Für selbst gesammeltes Zusatzmaterial: manuelles Sichten und Grob-Trimmen von Clips, Label per einfacher CSV/Ordnerstruktur (z. B. `data/raw/forehand/clip_003.mp4`) – kein spezialisiertes Tool nötig.
- Falls doch visuelle Prüfung/Korrektur der Pose-Erkennung gewünscht ist: **CVAT** (kostenlos, Open Source, lokal betreibbar) als Fallback-Tool, nicht als Pflichtwerkzeug.

**Aufwandsschätzung:** deutlich reduziert gegenüber Bounding-Box-Pipelines – geschätzt wenige Stunden für Sichten/Grob-Labeln, nicht Tage.

---

## 12. Model & Training Strategy

| Parameter | Wahl | Begründung |
|---|---|---|
| Pose-Modell | MediaPipe Pose (oder alternativ YOLOv8-Pose) | vortrainiert, kostenlos, CPU-tauglich, kein eigenes Training nötig |
| Classifier | MLP oder kleines 1D-CNN auf Sliding-Window-Keypoint-Features | wenige Parameter, schnell trainierbar, auch auf CPU machbar |
| Pretrained Weights | Pose-Modell: Standard-Gewichte des jeweiligen Frameworks | keine Fine-Tuning-Notwendigkeit für Pose selbst |
| Trainingsdaten | Kap. 10 | – |
| Fenstergröße | z. B. 15–25 Frames (~0.5–1s bei 25–30 FPS) | typische Schlagdauer, empirisch in Woche 2 zu verifizieren |
| Epochs | klein starten (z. B. 30–50), mit Early Stopping | Classifier ist klein, Overfitting-Risiko bei wenig Daten |
| Batch Size | 16–32 | Standardwert, an Datenmenge anpassen |
| Learning Rate | Start z. B. 1e-3 mit Scheduler/Reduktion bei Plateau | Standardvorgehen |
| Data Augmentation | Zeitliches Jittern des Fensters, leichtes Rauschen auf Keypoint-Koordinaten, horizontales Spiegeln (Vorsicht: vertauscht ggf. Forehand/Backhand bei Rechts-/Linkshändern – muss geprüft werden) | Robustheit bei kleinem Datensatz |
| Seed | fest gesetzt und dokumentiert | Reproduzierbarkeit |
| Hardware Training | primär CPU ausreichend für den kleinen Classifier; falls Pose-Feature-Extraktion für großen Datensatz zu langsam: Google Colab (kostenlose GPU-Kontingente) | keine bezahlten Lizenzen nötig |
| Trainingsdauer | erwartet im Minuten- bis niedrigen Stundenbereich | kleiner Classifier, kein großes Backbone-Training |

---

## 13. Baseline Experiment

**Gewählte Baseline:** *Einzelframe-Pose-Klassifikation (MVP-Ansatz)* vs. *Sliding-Window-Sequenz-Klassifikation (Target-Ansatz)*.

**Wissenschaftliche Aussage:** Zeigt quantitativ, ob und wie stark die Einbeziehung zeitlicher Information (mehrere Frames statt eines einzelnen) die Klassifikationsgüte verbessert – direkte empirische Beantwortung der in Kap. 6 des Auftrags gestellten konzeptionellen Frage "Einzelbild vs. temporale Betrachtung". Dieser Vergleich ist mit dem vorhandenen Zeitbudget realistisch, da beide Modelle auf denselben extrahierten Keypoint-Features aufbauen (kein zusätzlicher Trainingsaufwand für ein zweites komplettes Modell).

---

## 14. Evaluation & KPIs

| KPI | Was misst er | Warum relevant | Berechnung | Gutes Ergebnis (Hypothese) | Grenzen |
|---|---|---|---|---|---|
| Accuracy | Anteil korrekt klassifizierter Zeitfenster | Grundlegende Modellgüte | korrekt / gesamt auf Testset | – (erst nach Experiment zu benennen) | irreführend bei Klassenungleichgewicht |
| Precision/Recall/F1 pro Klasse | Klassenspezifische Güte | Zeigt z. B. ob Serve systematisch mit Forehand verwechselt wird | Standardformeln je Klasse | – | erfordert genug Testsamples pro Klasse |
| Confusion Matrix | Fehlermuster zwischen Klassen | Zeigt konkrete Schwächen für Diskussion/Demo | Kreuztabelle Vorhersage×Wahrheit | – | nur so aussagekräftig wie das Testset |
| Inference Latency | Zeit pro Frame/Fenster | Entscheidend für Echtzeit-Anspruch (Kap. 8 Grading) | gemessene ms pro Verarbeitungsschritt auf Ziel-Hardware | im niedrigen zweistelligen ms-Bereich pro Frame (Zielgröße, zu verifizieren) | hardwareabhängig, nicht übertragbar auf andere Geräte |
| FPS (End-to-End-Pipeline) | Verarbeitungsgeschwindigkeit des Gesamtsystems | Sichtbar in der Demo, direkte Interactivity-Relevanz | Frames verarbeitet / Sekunde | nahe Echtzeit (Zielgröße) | schwankt je nach Videoauflösung |
| Robustheit über Videos/Spieler | Performance-Streuung zwischen unterschiedlichen Quellvideos | Zeigt Generalisierung statt Überanpassung an ein Video | Accuracy je Video, Streuung (Std.) | möglichst geringe Streuung | nur so belastbar wie Anzahl unterschiedlicher Testvideos |
| Klassenverteilung im Datensatz | Anzahl Samples je Klasse | Grundlage zur Einordnung aller obigen Metriken | Zählung | ausgewogen genug, um F1 aussagekräftig zu machen | – |

Alle Zielwerte sind **Hypothesen**, keine bereits gemessenen Resultate (Kap. 32 des Auftrags).

---

## 15. Video Processing Pipeline

```
Video-Datei
   ↓
Frame-Extraktion (feste Sampling-Rate)
   ↓
Pose-Estimation pro Frame (MediaPipe/YOLOv8-Pose)
   ↓
Keypoint-Sequenz-Buffer (Sliding Window)
   ↓
Classifier → Klassenwahrscheinlichkeit pro Fenster
   ↓
Temporales Glätten / Debouncing (verhindert Flackern zwischen Klassen)
   ↓
Overlay-Rendering (Skelett + Label + Confidence) + Statistik-Update
   ↓
Interaktives Dashboard (Video-Player + Timeline + Statistik-Panel)
```

Läuft als lokaler Python-Prozess (OpenCV für Video-I/O, einfaches UI z. B. Streamlit oder ein leichtes Desktop-Fenster) – bewusst kein Server-/Cloud-Zwang für die Demo.

---

## 16. Interactivity Concept

Echte, nicht künstliche Interaktion, die den Grading-Kriterien "Real-Time/Feedback/Interaktion" gerecht wird:

- **Videoauswahl** aus vorbereitetem Set (Dropdown)
- **Start/Pause/Replay** der Analyse
- **Geschwindigkeitsregler** (0.5x/1x/2x) zur Demonstration der Robustheit über Tempo
- **Confidence-Threshold-Slider**, der live zeigt, wie sich erkannte Shots bei strengerer/lockerer Schwelle ändern (macht Modellverhalten *erlebbar*, nicht nur behauptet)
- **Klickbare Shot-Timeline**, die zum jeweiligen Zeitpunkt im Video springt
- **Live-Statistik-Panel** (Anzahl je Schlagart, aktualisiert während der Wiedergabe)

Diese Elemente sind in 4 Wochen mit Standard-Python-UI-Tools (Streamlit reicht aus) umsetzbar, laufen lokal, sind stabil und für ein gemischtes Fachpublikum in Sekunden verständlich.

---

## 17. Demo Concept

**Ablauf (Zielstruktur, wenige Minuten):**
1. Problem: "Amateurspieler bekommen selten objektives Schlag-Feedback."
2. Was macht KI: kurze Erklärung Pose-Estimation + gelernter Classifier (kein "die KI versteht Tennis").
3. Live: Video abspielen, Skelett-Overlay + Live-Klassifikation zeigen.
4. Interaktion: Threshold-Slider bewegen, Timeline anklicken, Statistik zeigen.
5. Ergebnis: Confusion Matrix / Accuracy kurz zeigen, ehrlich einordnen.
6. Warum interessant: Erweiterbarkeit (Success Rate, mehr Schlagarten) andeuten, ohne zu übertreiben.

**Demo Fallback:** vorab aufgezeichnete Analyse-Läufe (Video + bereits erzeugtes Overlay/Statistik als Screenshots/Recording) für den Fall, dass Live-Verarbeitung, Cloud oder Hardware während der Präsentation versagt.

---

## 18. GitHub Architecture

```
README.md              # Setup <1h, Architekturüberblick, Demo-Anleitung
requirements.txt
.gitignore              # schließt Video-/Modell-Rohdaten aus
src/
  pose_extraction.py
  windowing.py
  classifier.py         # Training + Inferenz
  pipeline.py           # End-to-End Video → Overlay
  dashboard.py           # interaktives UI
data/                   # nur Beispiel-/Demo-Clips, große Sets extern verlinkt
models/                 # trainierte Classifier-Gewichte (klein, ggf. per Git-LFS oder externem Download-Link)
scripts/
  download_data.py
  train.py
  evaluate.py
tests/
  test_pipeline.py
```

Große Rohdaten und ggf. das trainierte Modell liegen **nicht zwingend im Git-Repo**, sondern extern (z. B. Cloud-Speicher/Release-Asset) mit klar dokumentiertem Download-Link.

---

## 19. Reproducibility Strategy

Ziel: **Clone → Setup → Daten/Modell einbinden → Demo starten in < 1 Stunde.**

README dokumentiert eindeutig:
- Woher Daten stammen, wie sie heruntergeladen werden, welche Version/Commit verwendet wurde
- Wo das trainierte Modell liegt und wie es eingebunden wird (Downloadlink + Zielpfad)
- Exakte Setup-Schritte (venv/requirements.txt) inkl. geschätzter Laufzeit
- Wie die Demo gestartet wird (ein Befehl)
- Wie die Evaluation reproduziert wird (`scripts/evaluate.py`)

Training selbst muss laut Vorgabe nicht reproduziert werden, wird aber dennoch dokumentiert (Hyperparameter, Seed) für Nachvollziehbarkeit.

---

## 20. Ethics & Society (max. 1 A4-Seite)

**Kernrisiken:**
- Videoaufnahmen zeigen identifizierbare Personen (Spieler:innen) → Datenschutzrelevanz auch bei "nur" Bewegungsdaten.
- Missbrauchspotenzial: heimliche Leistungsüberwachung von Spieler:innen ohne Einwilligung; spätere Erweiterung Richtung Gesichtserkennung wäre grundsätzlich anders zu bewerten.
- Bias/Generalisierung: Trainingsdaten decken vermutlich nicht alle Spielstile, Körpertypen, Spielniveaus, Kameraperspektiven ab → Fehlklassifikationen bei unterrepräsentierten Gruppen wahrscheinlich.
- Fehlerhafte KI-Entscheidungen könnten – bei unreflektierter Nutzung – falsche Trainingsrückschlüsse befeuern.
- Mögliche kommerzielle Weiterverwendung ohne Zustimmung der ursprünglichen Video-/Datensatzquelle.

**Konkrete Gegenmaßnahmen:**
- Keine dauerhafte Speicherung von Videodaten während der Demo; nur vorab freigegebenes Demo-Material.
- Explizite Nennung der Datenquelle und deren Lizenz im Repo; keine Weiterverbreitung von Rohdaten ohne Berechtigung.
- Klare Kommunikation der Systemgrenzen (Kap. 21) in Doku und Präsentation, um Fehlinterpretation der Ergebnisse zu vermeiden.
- Keine Gesichtserkennung, keine Personenidentifikation im System.
- Hinweis in der Doku, dass ein produktiver Einsatz Einwilligung der gefilmten Personen voraussetzen würde.

---

## 21. Four-Week Roadmap

| Woche | Ziel | Kern-Tasks | Erwartetes Ergebnis | Std. (Hauptperson) | Exit Criteria | Risiken | Entscheidungspunkt |
|---|---|---|---|---|---|---|---|
| 1 | Technische Entscheidung + Datenquelle stehen | Architekturvergleich abschließen, Datenquelle(n) prüfen/downloaden, Pose-Modell testen, Setup Repo | Belastbare Architektur- und Datenentscheidung, erste Pose-Extraktion läuft | 20 | Pose-Estimation liefert auf 1 Testvideo plausible Keypoints | Datenquelle ungeeignet/Lizenzproblem | **Gate 1** |
| 2 | Dataset + erster Trainingslauf | Clip-Labeling, Feature-Extraktion (Keypoints) für gesamtes Set, Sliding-Window-Aufbau, MVP-Classifier trainieren | Trainierter MVP-Classifier mit erster Metrik | 20 | Testset-Accuracy klar über Zufallsniveau (>1/Klassenanzahl) | Klassenungleichgewicht, zu wenig Daten | **Gate 2** |
| 3 | Video-Pipeline + Demo + Evaluation | End-to-End-Pipeline (Video→Overlay), Dashboard/Interaktivität, Baseline-Vergleich, Robustheits-/Latenzmessung | Lauffähige Live-Demo auf Testvideos, vollständige KPI-Tabelle | 20 | Pipeline läuft stabil auf ≥3 unterschiedlichen Testvideos | Latenz zu hoch, Overlay instabil | **Gate 3** |
| 4 | Stabilisierung, Reproduzierbarkeit, Präsentation, Ethik | README/Repo finalisieren, Fallback-Demo aufzeichnen, Ethik-Reflexion schreiben, Präsentation proben | Reproduzierbares Repo (<1h Setup getestet von zweiter Person), fertige Demo inkl. Fallback | 20 | Externe Testperson kann Setup in <1h nachvollziehen | Zeitdruck kurz vor Präsentation | **Gate 4** |

---

## 22. Milestones (intern, keine offiziellen Zwischenabgaben)

- **M1 (Ende Woche 1):** Architekturentscheidung fixiert, Datensatz identifiziert und zugänglich.
- **M2 (Ende Woche 2):** MVP-Classifier trainiert und evaluiert.
- **M3 (Ende Woche 3):** Target-Demo (Video→Dashboard) funktionsfähig, Baseline-Vergleich vorliegend.
- **M4 (Ende Woche 4):** Reproduzierbares, präsentationsreifes Gesamtpaket inkl. Fallback.

---

## 23. Team Responsibilities

| Person | Rolle | Kritische Tasks? |
|---|---|---|
| Person 3 (Projektleiter, ~40% Erfahrung) | Architektur, Modell, Training, Integration, technische Leitung | Ja – zentral |
| Person 2 (~30% Erfahrung) | Dataset-Sichtung, Clip-Labeling, Testing, Dokumentation, unterstützende Entwicklung | Unterstützend, nicht auf kritischem Pfad |
| Person 1 (~0% Erfahrung) | Recherche zu Datenquellen, Datensammlung, einfaches Labeling, Präsentationsmaterial, manuelles Testing der Demo | Keine technisch kritischen Aufgaben |

Das Projekt ist so geplant, dass es auch funktioniert, wenn Person 1 und 2 weniger Zeit einbringen als vorgesehen – der kritische Pfad liegt vollständig bei Person 3.

---

## 24. Risk Register

| Risiko | Wahrscheinlichkeit | Auswirkung | Frühindikator | Gegenmaßnahme | Deadline |
|---|---|---|---|---|---|
| Zu wenig Daten | Mittel | Hoch | Woche 1: verfügbare Quellen zu klein | Zusätzliche Quelle einplanen, Augmentation verstärken | Ende Woche 1 |
| Schlechte/inkonsistente Labels | Mittel | Mittel | Stichprobenprüfung zeigt Fehler | Zweite Sichtung durch Person 2 | Ende Woche 2 |
| Klassenungleichgewicht | Hoch | Mittel | Klassenverteilung nach Sammlung geprüft | Class Weights im Training, ggf. gezielt Serve-Clips nachsammeln | Ende Woche 2 |
| Schlechte Modellperformance | Mittel | Hoch | MVP-Accuracy nahe Zufallsniveau | Fenstergröße/Feature-Set anpassen, Architektur vereinfachen (Gate 2) | Ende Woche 2 |
| Overfitting | Mittel | Mittel | große Lücke Train- vs. Val-Accuracy | Early Stopping, mehr Augmentation, Regularisierung | laufend |
| Data Leakage | Niedrig (bei sauberem Split) | Hoch | unrealistisch hohe Testperformance | Split strikt auf Video-/Spielerebene erzwingen und prüfen | Ende Woche 2 |
| Zu hohe Inferenzlatenz | Mittel | Mittel | FPS-Messung Woche 3 unter Zielwert | Sampling-Rate reduzieren, Fenstergröße/Complexity senken | Ende Woche 3 |
| Cloud-GPU nicht verfügbar | Niedrig (Ansatz ist CPU-fähig) | Niedrig | Colab-Kontingent erschöpft | Lokales CPU-Training als Fallback (Classifier ist klein genug) | laufend |
| Zeitüberschreitung | Mittel | Hoch | Task dauert >150% der Schätzung | Gates konsequent nutzen, Scope kürzen statt Timeline strecken | laufend |
| Videoqualität/Perspektivwechsel | Mittel | Mittel | Robustheitsmessung Woche 3 zeigt hohe Streuung | Robustheit explizit als Limitation kommunizieren statt verschweigen | Ende Woche 3 |
| Interactivity funktioniert nicht stabil | Niedrig–Mittel | Hoch (Grading-relevant) | UI-Tests Woche 3/4 zeigen Abstürze | Fallback: statische, vorab erzeugte Interaktionsdemo | Ende Woche 4 |
| Demo-Ausfall am Präsentationstag | Niedrig | Hoch | – | Aufgezeichnete Fallback-Demo (Kap. 17) | Bereit ab Ende Woche 4 |
| Ethische/rechtliche Probleme (Lizenz/Datenschutz) | Niedrig | Hoch | Lizenzprüfung Woche 1 unklar | Nur klar lizenzierte Quellen verwenden, im Zweifel Quelle wechseln | Ende Woche 1 |

---

## 25. Fallback Strategy

- **Gate 1 (Daten/Annotation scheitert):** alternative Datenquelle bzw. reduzierter Klassenumfang.
- **Gate 2 (Modell erreicht keine brauchbare Performance):** Architektur vereinfachen (z. B. nur MLP statt 1D-CNN, kleineres Fenster).
- **Gate 3 (Video-Pipeline instabil):** MVP + vorbereitete, bereits verarbeitete Videos als "Target"-Ersatz in der Demo.
- **Gate 4 (Stretch Goal – temporale Erweiterung/Success Rate – zu aufwändig):** ersatzlos streichen, MVP+Target bleiben vollständig.
- **Demo-Tag:** aufgezeichnete Fallback-Demo für Netzwerk-/Hardwareausfall.

---

## 26. Definition of Done

**MVP:** Classifier auf getrenntem, videobasiert gesplittetem Testset evaluiert; Accuracy, Precision/Recall/F1 und Confusion Matrix liegen dokumentiert vor; Ergebnis reproduzierbar über `scripts/evaluate.py`.

**Target:** End-to-End-Pipeline verarbeitet ein beliebiges Video aus dem Demo-Set fehlerfrei bis zum Dashboard; alle in Kap. 16 gelisteten Interaktionselemente sind funktionsfähig; FPS/Latenz sind gemessen und dokumentiert.

**Stretch Goal:** Falls umgesetzt – Vergleichsmetrik zwischen Baseline (MVP) und temporaler Erweiterung liegt vor; falls Success Rate umgesetzt – Definition und Messmethode sind explizit dokumentiert, inkl. Limitationen.

**Final Project:** Repository von einer projektfremden Person in <1h gemäß README aufgesetzt und Demo erfolgreich gestartet; Ethik-Reflexion (≤1 Seite) vorhanden; Präsentationsskript und Fallback-Demo vorbereitet.

---

## 27. Presentation Strategy

Wording konsequent präzise statt werblich:
- **Verwenden:** "AI-assisted tennis shot recognition", "pose-based movement classification", "erkannte Schlagarten mit gemessener Genauigkeit".
- **Vermeiden:** "die KI versteht Tennis", "vollautomatisches Coaching", unbelegte "Echtzeit"-Aussagen ohne gemessene FPS/Latenzwerte.

"AI Tennis Coach" wird als übergeordnetes Anwendungskonzept verwendet, mit expliziter Klarstellung, dass das aktuelle System Shot Recognition und Performance-Übersicht leistet – kein Coaching im eigentlichen Sinn.

---

## 28. Expected Grading Assessment

| Kategorie | Erwartetes Niveau | Begründung | Größte verbleibende Schwäche | Konkrete Maßnahme |
|---|---|---|---|---|
| Concept & Creativity | Gut–sehr gut | Realistischer, ehrlich abgegrenzter Use Case | Konzept an sich nicht neuartig | In Präsentation klaren Mehrwert (objektive, wiederholbare Analyse) betonen |
| Technical Implementation & Interactivity | Gut | Nahezu Echtzeit auf CPU, mehrere echte Interaktionselemente | Keine echte Live-Webcam-Demo | Bewusst begründen (Raumsituation), Video-Interaktivität als gleichwertig positionieren |
| AI Model & Technical Grounding | Gut–sehr gut | Begründete Architekturwahl, eigener Trainingslauf, Baseline-Vergleich | Kleiner, evtl. nicht perfekt ausbalancierter Datensatz | Klassenverteilung früh prüfen, Class Weighting einsetzen |
| Documentation & Reproducibility | Sehr gut (Zielsetzung) | Klar geplante Repo-Struktur, <1h-Setup als hartes Kriterium | Zeitdruck in Woche 4 kann Doku-Qualität gefährden | Doku parallel zur Entwicklung schreiben, nicht erst am Ende |
| Ethical & Societal Reflection | Gut | Konkrete Risiken und Gegenmaßnahmen statt generischer Floskeln | Begrenzung auf 1 Seite erzwingt Kürze | Fokus auf die 2–3 relevantesten Risiken statt Vollständigkeit |
| Presentation & Communication | Gut–sehr gut | Klares 6-Punkte-Skript, Fallback vorbereitet | Ungeübtes Team bei technischen Fragen aus dem Publikum | Kritische Fragen (Kap. 29) vorab durchspielen |

---

## 29. Kritische Gegenprüfung ("Warum keine Bestnote?")

**5 technische Schwächen + Gegenmaßnahme**
1. Pose-Estimation kann bei schneller Bewegung/Bewegungsunschärfe versagen → Frame-Sampling-Rate und Videoqualität in Datenauswahl berücksichtigen, Grenzen offen kommunizieren.
2. Kleiner, evtl. unausgeglichener Datensatz → Class Weighting, ehrliche Diskussion der Limitation in der Präsentation.
3. Sliding-Window-Ansatz erkennt Übergänge zwischen Schlägen evtl. unscharf → Debouncing/Glättung, als bekannte Grenze dokumentieren.
4. Kein echtes Ballverfolgungs-/Kontextmodell → explizit als bewusste Scope-Entscheidung begründen, nicht verschweigen.
5. Generalisierung auf völlig neue Kameraperspektiven ungetestet → Robustheits-KPI misst dies explizit und wird transparent berichtet.

**5 Projektrisiken + Gegenmaßnahme**
1. Zeitüberschreitung bei Datensammlung → harte Zeitboxen + Gate 1.
2. Überambitionierte Stretch Goals gefährden MVP → Priorisierungsprinzip (Kap. 30 Auftrag) konsequent durchsetzen.
3. Alleinige Abhängigkeit von Person 3 → frühzeitige Wissensteilung, dokumentierter Code.
4. Lizenzunsicherheit bei Videomaterial → Lizenzprüfung fest in Woche 1 verankert.
5. Demo-Technik-Ausfall am Präsentationstag → aufgezeichnete Fallback-Demo Pflichtbestandteil.

**5 mögliche Expertenfragen + Antwortstrategie**
1. *"Warum kein End-to-End-Deep-Learning-Modell auf Rohvideo?"* → Antwort: bewusste Abwägung Datenmenge/Hardware/Zeit vs. Ansatz B (Kap. 5/6).
2. *"Wie sicher ist eure Klassenzuordnung bei mehrdeutigen Bewegungen?"* → Confusion Matrix und Confidence-Threshold-Demo direkt zeigen.
3. *"Wie geht ihr mit Data Leakage um?"* → Split-Strategie auf Video-/Spielerebene erklären (Kap. 10).
4. *"Ist das wirklich Echtzeit?"* → gemessene FPS/Latenzwerte präsentieren, keine Behauptung ohne Zahl.
5. *"Was passiert bei einem neuen, komplett anderen Video?"* → Robustheits-KPI und offen kommunizierte Generalisierungsgrenze zeigen.

---

## Finale Entscheidung

> **"Wenn wir nur vier Wochen und maximal 20 Stunden pro Woche für die Hauptperson haben, würden wir dieses Projekt genau so umsetzen:"**

1. **Architektur:** Pose Estimation (vortrainiert) + leichter Sequenz-Classifier (MLP/1D-CNN) über Sliding-Window-Keypoint-Features.
2. **Modell:** MediaPipe Pose (Alternative: YOLOv8-Pose) für Keypoints; selbst trainierter, kleiner Classifier für Forehand/Backhand/Serve/kein Schlag.
3. **Dataset:** öffentlich verfügbarer, bereits gelabelter Tennis-Action-Datensatz (Lizenz in Woche 1 verifizieren) plus wenige selbst zugeschnittene Clips zur Robustheitsprüfung; Split strikt auf Video-/Spielerebene.
4. **Annotation:** nur Clip-/Fenster-Labels, keine Bounding Boxes; Aufwand bewusst niedrig gehalten.
5. **Training:** kleiner Classifier, CPU-tauglich, optional Colab; fester Seed, dokumentierte Hyperparameter, Early Stopping.
6. **Evaluation:** Accuracy/Precision/Recall/F1/Confusion Matrix + Latenz/FPS + Robustheit über mehrere Videos; MVP-vs.-Target-Baseline-Vergleich.
7. **Video Pipeline:** Frame-Extraktion → Pose → Sliding Window → Classifier → Glättung → Overlay, lokal auf dem Laptop lauffähig.
8. **Interactivity:** Videoauswahl, Start/Pause/Replay, Geschwindigkeit, Confidence-Threshold-Slider, klickbare Shot-Timeline, Live-Statistik-Panel.
9. **Demo:** video-basiert (kein Live-Webcam-Zwang), 6-Punkte-Skript, mit aufgezeichnetem Fallback.
10. **MVP:** trainierter Classifier mit gemessenen Metriken auf getrenntem Testset, unabhängig von der vollen Pipeline lauffähig.
11. **Stretch Goal:** echte temporale Erweiterung (LSTM/Temporal-CNN) und/oder ehrlich definierte Shot Success Rate – nur wenn MVP/Target vollständig stehen.
12. **Fallback:** bei jedem Gate-Scheitern Vereinfachung statt Verkomplizierung; aufgezeichnete Demo als letzte Absicherung.
