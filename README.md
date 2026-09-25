# GenWorld

Generatore di mondi Minecraft (Java Edition) a partire da piante, disegni,
heightmap e dati geografici reali.

Stato: **funziona e si gioca.** Da un'immagine disegnata si ottiene un mondo
Minecraft giocabile - terreno, fiumi, vulcani, biomi, vegetazione, citta' con
mura e mercato, botteghe con gli artigiani dentro, campi coltivati, strade e
ponti - aperto e provato in gioco sulla 1.21.4. C'e' anche la finestra.

## L'idea in una riga

La scala del **terreno** e la scala degli **oggetti** sono indipendenti. Il
terreno si comprime per far stare un territorio grande in pochi blocchi; case,
ponti e monumenti restano a dimensione giocabile. Il rapporto fra le due e' il
**fattore di esagerazione**, e da li' discende tutto il resto.

Niente case mignon: il generatore di edifici non ha accesso alla scala del
terreno, quindi non puo' sbagliare.

## Installazione

Lo **stimatore di fedelta'** non ha dipendenze: basta Python 3.10+.

Per **scrivere mondi** serve `amulet-core`, che pero' richiede `numpy<2` e
rompe altri pacchetti se installato nell'ambiente di sistema. Va in un venv
dedicato, **fuori dalla cartella Google Drive** (un venv sono migliaia di file
e li sincronizzerebbe tutti):

```bat
py -m venv C:\venvs\genworld
C:\venvs\genworld\Scripts\pip install amulet-core pillow scipy PySide6-Essentials
```

Poi si lancia con quell'interprete:

```bat
C:\venvs\genworld\Scripts\python esempi\spike_piatto.py
```

## Uso

### La finestra

Doppio clic su **`GenWorld.bat`**, nella cartella del progetto.

Al primo avvio crea l'ambiente in `C:\venvs\genworld` e installa le
dipendenze (qualche minuto, una volta sola); dalle volte dopo parte e basta.
Controlla anche che numpy sia rimasto alla serie 1: amulet non regge la 2, e
un pacchetto presente ma rotto non si vede guardando l'elenco di pip.

Equivale a:

```bat
C:\venvs\genworld\Scripts\python -m genworld.gui
```

Un vero `.exe` si puo' costruire solo su Windows, quindi in cartella c'e' la
ricetta invece del risultato: **`crea_exe.bat`** lo impacchetta con
PyInstaller. E' sperimentale - amulet si porta dietro i dati di PyMCTranslate,
che PyInstaller tende a lasciare indietro - e il `.bat` resta la strada che
funziona sempre.

![la finestra](mondi/gui.png)

E' il punto da cui il progetto era partito: "un applicativo per pc". A
sinistra la mappa, il lato in blocchi e i cursori di cosa generare; a destra
l'anteprima (classi, quote dedotte, mondo finito) e il **preventivo di
fedelta' che si ricalcola mentre si trascinano i cursori**.

Il preventivo e' un preventivo, non un voto: una riga per dimensione, perche'
un numero unico nasconderebbe proprio la cosa che interessa, *cosa* si perde.
E non c'e' nessun controllo per la scala degli oggetti, che resta 1 m per
blocco: il cursore muove la scala del **terreno**, e il pannello mostra il
fattore di esagerazione che ne risulta. Una casa resta una casa in cui si
entra anche su un mondo da 256 blocchi.

La generazione gira in un thread: la barra avanza, il bottone Annulla
funziona, e chiudere la finestra a meta' lavoro non lascia un mondo monco.

### Stimatore di fedelta'

```bash
python -m genworld.cli stima --preset garda --lato 2048 --auto-verticale
python -m genworld.cli confronta --preset italia --auto-verticale
python -m genworld.cli risolvi --preset italia --errore 500        # fisso la fedelta'
python -m genworld.cli risolvi --preset garda --metri-per-blocco 1 # fisso la scala
python -m genworld.cli risolvi --preset italia --lato 1000         # fisso il mondo
```

Preset: `italia`, `garda`, `valorcia`, `etna`.
Monumenti: `torre_eiffel`, `colosseo`, `duomo_milano`, `torre_pisa`, `mole`,
`burj_khalifa`.

### Calibrare la lettura di una mappa

Su cinque mappe reali di stili diversi la classificazione automatica funziona su
due e fallisce su due. I modi di fallire non sono tarature da aggiustare: sono
cose che una persona vede in un secondo e un algoritmo no - una cornice
illustrata, un riquadro di legenda, un mare dipinto del colore della terra.

Quindi si chiedono due gesti, con lo strumento **Calibratore Mappe** (pagina web,
l'immagine resta nel browser):

1. un **rettangolo** attorno alla mappa vera - elimina cornici, legende,
   cartigli e rose dei venti in un colpo solo;
2. qualche **clic col contagocce** - "questo e' mare, questo e' montagna,
   questo e' decorazione da ignorare";
3. se serve, qualche **pennellata** terra/acqua - la via di fuga per le mappe
   che nessuna regola sa leggere.

Le pennellate decidono solo l'appartenenza a terra o acqua, non la classe:
dove concordano col colore resta il dettaglio della classificazione. Viaggiano
dentro il JSON del profilo come maschera 192x192 compressa a lunghezze di corsa
(~1,5 KB), quindi non c'e' un file a parte da tenere allineato.

Si copia il profilo JSON, si salva accanto alla mappa e si genera:

```bash
python esempi/genera_mappa.py --profilo input/ansalon.profilo.json --lato 832 --lotto 0
```

Misurato su cinque mappe reali di stili diversi: **da due su cinque a quattro
su cinque**. Su Ansalon (seppia con cornice illustrata, il caso peggiore)
l'acqua passa dall'**1,5% al 46,7%** e il decoro viene azzerato dal rettangolo,
ma la geografia resta imprecisa: li' serve il pennello.

Effetto collaterale utile: ritagliata la cornice, anche `famiglia()` smette di
sbagliare - la cornice illustrata gonfiava la grana e faceva passare una mappa
dipinta per fotografica.

### Da una mappa disegnata a un mondo

```bash
# una mappa grande si genera a lotti: il lotto 0 crea, i successivi aggiungono
python esempi/genera_mappa.py --lato 832 --lotto 0 --chunk-per-lotto 700
python esempi/genera_mappa.py --lotto 1 --chunk-per-lotto 700
python esempi/genera_mappa.py --lotto 2 --chunk-per-lotto 700
python esempi/genera_mappa.py --lotto 3 --chunk-per-lotto 700
python esempi/genera_mappa.py --anteprima
```

Produce `mondi/arda/` (832x832 blocchi, 2704 chunk, 11 MB) e
`mondi/arda_anteprima.png`. Senza `--chunk-per-lotto` fa tutto in una volta.

La pipeline vera sta in `genworld/motore.py`: `esempi/genera_mappa.py` e la
finestra sono due interfacce sopra le stesse funzioni (`analizza`,
`pianifica`, `scrivi`), cosi' non possono divergere.

Opzioni principali:

```
--profilo FILE     profilo del Calibratore (rettangolo + campioni + pennello)
--fiumi SOGLIA     area drenata minima per un fiume (150; 0 = nessuno)
--alberi SCALA     densita' della vegetazione (1,0; 0 = nessuna)
--villaggi SCALA   densita' degli insediamenti (1,0; 0 = nessuno)
--erosione GOCCE   erosione idraulica (0 = spenta; utile sui terreni lisci)
--vulcani N        quanti coni vulcanici piazzare (3; 0 = nessuno)
--senza-strade     non tracciare la rete stradale
```

Su Arda: 3 vulcani (19.418 celle di cono, 767 di cratere, 423 di lago di lava,
1.114 di colata, cima a quota 143), 1.363 celle di fiume in 26 corsi, ~3.800
alberi, 91 edifici in 5 centri (3 murati), 3.811 celle di sede stradale di cui
2.122 lastricate, e 164 di ponte.

I ponti non si piazzano: emergono. Ogni tratto di strada che finisce sull'acqua
o su un avvallamento profondo diventa ponte, perche' e' l'unico modo di stare
alla quota della sede. Uno di quelli su Arda attraversa uno stretto per 116
celle - nessuno l'ha chiesto, la strada doveva arrivare a quel villaggio.

L'ordine della pianificazione non e' negoziabile, perche' tre fasi modificano
il terreno: insediamenti (spianano i lotti), strade (spianano la sede),
vegetazione (si semina su quel che resta). Nella scrittura di ogni chunk:
terreno, alberi, edifici, strade.

Quattro trappole delle mappe dipinte, tutte risolte in `genworld/mappa.py` e
documentate nei doc di progetto: l'ombreggiatura dentro i colori (si classifica
in HSV sulla tinta), la stessa ombreggiatura come segnale (contrasto locale =
rilievo), le scritte che diventano montagne a forma di parola (maschera **e**
rugosita' azzerata), il bordo dell'immagine che diventa una muraglia (ritaglio
**e** margine azzerato).

### Generazione di un mondo di prova

```bash
python esempi/spike_piatto.py
```

Produce `mondi/spike/` (mondo Minecraft 256x256) e `mondi/spike_anteprima.png`.
Con `--liscio` genera la variante senza rumore, per confronto.

Per provarlo in gioco, copiare la cartella `mondi/spike` dentro
`%APPDATA%\.minecraft\saves\`.

### Test

```bash
python -m pytest tests -q                                   # 163, 33 saltati
C:\venvs\genworld\Scripts\python -m pytest tests -q          # 196 test
```

I test saltati senza il venv sono quelli che richiedono amulet (scrittura dei
mondi) o PySide6 (la finestra). L'aspetto della finestra non si prova con un
test: si guarda, e per guardarlo senza schermo c'e'
`python esempi/scatto_gui.py --analizza`, che la disegna offscreen e la
fotografa in `mondi/gui.png`.

## Cosa ha trovato lo spike

**amulet-core scrive i file region correttamente ma produce un `level.dat`
monco**: contiene solo `DataVersion`, `LastPlayed`, `LevelName` e `version`.
Mancano `WorldGenSettings`, `GameType`, `Version`, lo spawn e i gamerule, e
all'apertura amulet stesso segnala che i template delle dimensioni sono
malformati. Minecraft rifiuterebbe il mondo o lo rigenererebbe.

Soluzione: `genworld/livello_dat.py` ricostruisce il `level.dat` da zero dopo
la creazione, con `verifica()` che controlla la presenza e il tipo di tutti i
campi che Minecraft legge all'apertura.

Scelta collegata: l'overworld e' un **superflat di sola aria**. Il terreno lo
scriviamo noi, e cosi' i chunk che non tocchiamo restano vuoti invece di
riempirsi di terreno vanilla a caso ai bordi della mappa.

**Trappola del renderer**: il palette dei blocchi di un livello amulet e' vuoto
appena caricato e si popola solo man mano che si leggono i chunk. Costruire la
mappa colori prima del ciclo da' un'immagine tutta ignota. Va risolto
pigramente, dentro il ciclo.

**DataVersion: non scriverla a mano.** Una tabella compilata a memoria aveva
gia' due valori sbagliati di un'unita' (1.21.5 e 1.21.8). Ora
`data_version_di()` la chiede a PyMCTranslate, che arriva con amulet ed e' la
fonte autorevole.

## Il terrazzamento

Aperto il mondo in gioco, il versante mostrava curve di livello a gradini. Non
e' un difetto del rilievo: e' un artefatto della **quantizzazione** a blocchi
interi, e una superficie liscia che attraversa lentamente il confine fra due
interi lo attraversa lungo una curva di livello.

Sommare fBm prima di arrotondare non basta: le ottave fini portano
un'ampiezza `persistenza^n`, cioe' centesimi di blocco, mentre serve
variazione dell'ordine del blocco proprio alle frequenze alte.

La soluzione e' ditherare la soglia di arrotondamento (`rumore.quantizza`):
costa 0,03 blocchi di scarto medio e i gradini spariscono. Confronto visivo in
`mondi/confronto_terrazze.png`.

Nota: quattro metriche automatiche di terrazzamento sono state provate e
scartate, nessuna discrimina. Dettagli in `genworld/rumore.py`.

## Verifica

Lo spike non e' "sembra giusto": 256x256 blocchi generati, riletti dai file
region e confrontati colonna per colonna con la heightmap di partenza.

```
colonne confrontate: 65536
differenze diverse da zero: 0
errore max: 0
```

Resta un solo passaggio che qui non si puo' fare: aprire il mondo in Minecraft.
La struttura del `level.dat` e' validata campo per campo, ma la prova definitiva
e' il doppio clic.

## I vulcani e la ruota con i raggi

Il cono e' l'unico elemento che si **sovrappone** al terreno invece di
dedurlo: si somma alla heightmap con `np.maximum`, cosi' un vulcano su una
collina ne eredita la base. Il profilo e' concavo verso l'alto
(`1 - t**1.5`), il cratere e' un invaso con il fondo piatto e la lava sta al
suo livello, non a quello del mare.

Il primo risultato era geometricamente corretto e visivamente falso: le colate
scendevano come i raggi di una ruota. Un cono matematicamente perfetto ha il
gradiente **esattamente radiale**, quindi la massima pendenza e' la linea
retta dal cratere.

Il primo rimedio - un fBm planare sommato al cono - non ha cambiato nulla:
con celle piu' larghe del cono il rumore lo **inclina** soltanto. Il rumore
deve essere **angolare**: la quota dipende dall'angolo attorno al cratere,
cosi' le pieghe corrono lungo la linea di massima pendenza e diventano
valloni. Sopra ci vanno altre due correzioni, perche' tre armoniche pure
danno una zucca scanalata: un'onda lenta che apre alcuni settori e ne
appiattisce altri, e un fBm fine che sporca i crinali. Il raggio stesso e'
deformato del 10% da tre onde lente, altrimenti la base e' un cerchio di
compasso.

Valore scelto `rugosita = 0,18`, dal confronto in `mondi/vulcani.png`
(`python esempi/prova_vulcani.py 0 0.12 0.18 0.26`).

Difetto collegato, trovato dallo stesso confronto: senza una soglia di
pendenza l'inerzia della colata tirava righe rette per mezza mappa, perche' in
pianura il gradiente e' nullo e nulla la frena. Ora la lava si ferma dove il
pendio si spiana - si raffredda, non arriva al mare.

E un bug vero, scoperto da un test: in `scegli_siti` un sito scartato perche'
troppo vicino al bordo consumava comunque uno dei vulcani richiesti, e su una
mappa con poca terra utile se ne otteneva **zero**.

## I biomi, ovvero il clima

Per molto tempo il mondo ha avuto i blocchi giusti e nessun clima: ogni chunk
usciva `plains`. In gioco vuol dire erba dello stesso verde dal deserto alla
tundra, acqua di un colore solo, niente neve che si posa, mob sbagliati. I
biomi sono l'unica cosa che Minecraft **non** deduce dai blocchi.

Il clima si calcola, non si dichiara: un campo di "freddo" da 0 a 1, tagliato
in tre fasce. Due contributi:

- la **quota**, che e' il contributo certo - l'aria si raffredda salendo e la
  heightmap ce l'abbiamo;
- la **latitudine**, ma solo se il disegno la suggerisce. Una mappa disegnata
  non dice dove sia il nord, percio' invece di inventare un asse si guarda
  dove sta la classe NEVE: se e' concentrata da una parte, quella parte e' il
  freddo. Se e' al centro non e' latitudine, e' quota; se non c'e', nessun
  gradiente.

Sopra a tutto un fBm largo, perche' una soglia netta segue una curva di
livello e un confine di bioma che segue una curva di livello si riconosce a
colpo d'occhio come generato.

Due errori, uno di misura e uno di lettura.

**La latitudine andava centrata.** Presa cosi' com'e', da 0 a 1, metteva
meta' mappa sotto la soglia del caldo: l'oceano tropicale diventava il bioma
piu' esteso del mondo, 154.615 celle su Arda, piu' dell'oceano normale. Un
emisfero intero non puo' essere "l'estremo caldo". Ora e' uno scostamento da
un clima temperato.

**Un nome moderno non e' un nome universale.** `universal_minecraft:snowy_plains`
non esiste - l'universale e' `snowy_tundra` - e `from_universal` su un nome
sconosciuto lo restituisce tale e quale, senza un errore. E' esattamente la
trappola di `universal_minecraft:oak_log`, la seconda volta. Stavolta pero'
c'e' una guardia: `biomi.verifica_traduzioni()` prova tutta la tavolozza
contro PyMCTranslate, e un test la chiama.

Verifica finale con il metodo dello spike: mondo generato, file region
riaperti, biomi riletti e confrontati con la mappa che li ha prodotti. Zero
differenze. Su Arda: 27 biomi distinti, `mondi/biomi.png` mette a confronto
le classi e i biomi.

## Il difetto che solo il gioco poteva mostrare

Aperto il mondo in Minecraft 1.21.4: terreno giusto, alberi giusti, biomi
giusti - e in sovrimpressione, a ripetizione:

    Caricamento del chunk in [-11, -6] non riuscito
    Controlla il registro per ulteriori dettagli

Nel registro, 1.193 volte in una partita:

    Failed to parse chunk [x, z] position info
    java.lang.ArrayIndexOutOfBoundsException: Index 0 out of bounds for length 0
    Failed to load chunk x,z

Non erano i file region. Quelli erano stati riletti e confrontati piu' volte
- colonna per colonna per le quote, cella per cella per i biomi - ed erano a
posto. Era la cartella **`entities/`**, che amulet scrive accanto a
`region/`: per ogni chunk un chunk-entita' di 22 byte contenente solo
`DataVersion: 0`, senza `Position` e senza `Entities`. Minecraft legge la
posizione da quell'array, trova un array vuoto, e va fuori indice.

I due archivi sono separati, quindi il terreno si caricava lo stesso: il
difetto era invisibile a tutto quello che avevamo controllato, perche'
avevamo controllato solo la meta' a cui stavamo pensando.

I nostri mondi non contengono entita', quindi la cartella giusta e' nessuna
cartella: `ScrittoreMondo` la elimina alla chiusura e Minecraft la ricrea
quando serve. Il mondo passa da 22 a 11 MB.

Nello stesso registro, un secondo avviso minore: `key missing: DragonFight`.
Minecraft lo cerca nel level.dat anche se l'End non l'hai mai visto. Ora c'e'.

`tests/test_mondo_valido.py` rilegge i file **senza passare da amulet** - un
controllo fatto con lo stesso strumento che ha scritto il file non e' un
controllo - e pretende quello che pretende Minecraft: `xPos`, `zPos`, `yPos`,
`Status`, `sections`, `Heightmaps` in ogni chunk, `Position` in ogni
chunk-entita' se la cartella esiste, e un level.dat senza lamentele.

## Le citta'

Prima un "villaggio" era una griglia sfalsata di case dentro un cerchio.
Dall'alto si riconosceva subito: le case non guardavano niente, non c'era un
dentro e un fuori, e lo spazio fra loro non era uno spazio, era l'avanzo.

Una citta' e' il contrario: prima esiste il **vuoto** - piazza, strade,
vicoli - e le case vengono dopo, appoggiate a quel vuoto.

**Impianto organico, non a griglia.** Le strade non si disegnano: sono i
collegamenti fra i luoghi. Si spargono dei punti nell'area urbana, si
triangolano con Delaunay, e le vie sono gli spigoli di quella
triangolazione - una maglia irregolare, piena di isolati chiusi di forma
diversa, che e' il modo in cui cresce un borgo. La gerarchia si deduce e non
si dichiara: sono **assi** gli spigoli sul cammino piu' breve fra una porta e
la piazza, **secondarie** quelli lunghi, **vicoli** tutto il resto.

**Dove sta il nord della citta'.** Il perimetro e' deformato da tre onde
lente (un perimetro circolare si legge come un timbro) e tagliato su cio' che
il terreno concede. La piazza non sta al centro geometrico ma nel punto piu'
piano e piu' interno: su un sito tagliato da un fiume il centro geometrico
puo' cadere in acqua.

**Il lotto si cerca, non si propone.** Il primo tentativo sceglieva una
misura a caso e la scartava se non entrava: 288 posizioni buttate su 830, e
otto case costruite in una citta' intera. Ora il rettangolo **cresce** da una
cella sul fronte strada - prima in profondita' fin dove l'isolato lo lascia
andare, poi di lato - ed e' esattamente quello che fa una casa a schiera. Da
8 case a 34. E' anche il motivo per cui nei centri storici i lotti sono
stretti, lunghi e tutti diversi.

Poi c'e' il gradiente: al centro le case si toccano e hanno due o tre piani,
in periferia sono staccate e basse. Senza, e' un quartiere residenziale
caduto dal cielo.

**Le porte non si piazzano.** Il primo tentativo cercava le celle di cinta
gia' toccate da una strada e ne trovava zero - le strade finiscono
nell'abitato, la cinta gira fuori. Ora e' la via che viene prolungata fino al
muro, che e' anche quello che succede davvero: la porta esiste perche' ci
passa la strada, non viceversa. E la strada esterna punta alla porta, non al
centro: prima entrava in citta' e passava in mezzo alle case.

**I tetti si sovrapponevano**, e si vedeva solo dall'alto. Il sedime di due
case adiacenti non si tocca mai - c'e' un test che lo controlla dall'inizio -
ma il tetto sporge di un blocco oltre i muri, e due gronde in un vicolo
stretto finiscono nella stessa cella. L'ingombro vero di una casa vista
dall'alto non e' il sedime, e' il sedime **piu' la gronda**.

Pretendere lo spazio per la gronda pero' faceva scendere Arda da 91 edifici a
59: un terzo del paese demolito per un blocco di sporgenza. La regola giusta
non e' rifiutare la casa, e' rinunciare alla gronda - che e' esattamente
quello che fa una casa a schiera. Ora chi ha un vicino attaccato ha il tetto
a filo di muro, chi ha spazio se la tiene.

**Dentro le case** ora c'e' qualcosa: letto lontano dalla porta, focolare e
banco contro il muro, cassa, scala a pioli sotto il buco nel solaio, torce.
Non si vede dall'alto, ma entrarci era entrare in una scatola.

Su Arda: 91 edifici in 5 centri, 3 con le mura, 2.807 celle di via interna,
808 di cinta, 45 di porta.

## Mestieri, botteghe e abitanti

Una citta' di sole case e' un dormitorio. Ogni edificio ora puo' avere un
**mestiere**, dedotto da dove sta: le botteghe sulla piazza e sulle vie
principali (libraio, speziale, cartografo, macellaio, fruttivendolo, fabbro),
gli artigiani rumorosi un po' piu' in la' (armaiolo, corazzaio, falegname,
scalpellino, conciatore), e chi lavora la terra o il pesce al bordo, vicino a
quello che lavora. E' la ragione per cui i mestieri hanno dato i nomi alle vie
di mezza Europa.

Dentro, ogni bottega ha il suo banco - incudine e fucina dal fabbro,
affumicatore dal macellaio, compostiera e ceste dal fruttivendolo, telaio dal
pastore - e il banco non e' decorazione: in Minecraft e' il "posto di lavoro"
che un abitante rivendica.

Sulla piazza si apre il **mercato**: banchi 3x3 con quattro pali, la tenda
colorata e il bancone rivolto verso la piazza, con la merce sopra. Frutta,
carne, pesce, verdura.

### Gli abitanti, e il secondo file che amulet scrive male

Nelle case e dietro i banchi ci sono i villager. Metterceli ha richiesto di
scrivere a mano i file `entities/*.mca`: **amulet non serializza le entita'**.
Si puo' riempire `chunk.entities` di oggetti perfettamente formati e quello
che finisce sul disco resta il solito chunk da 22 byte con dentro
`DataVersion: 0` - lo stesso file che faceva scorrere all'infinito
"Caricamento del chunk non riuscito".

Quindi `genworld/entita.py` scrive il formato Anvil da zero: intestazione da
8 KiB, settori da 4096 byte, NBT compresso con zlib, e dentro `Position` ed
`Entities` come li vuole Minecraft. E' la seconda volta che tocca rifare cio'
che amulet scrive male - la prima era il `level.dat`.

Difetto trovato subito dopo, e istruttivo: i primi abitanti sono finiti nelle
regioni `r.0.0`/`r.1.1` invece di `r.-1.-1`/`r.0.0`. Tutta la pianificazione
lavora in coordinate di **mappa** (0..lato), il mondo e' centrato
sull'origine, e i blocchi lo sanno perche' glielo dice chi li scrive; le
entita' no. Nessun errore, nessun avviso: trentadue abitanti scritti in un
angolo di mondo vuoto.

Su Arda: 94 edifici, 32 botteghe, 12 banchi di mercato, 44 abitanti.

## La campagna

Un paese senza campi e' un fondale: le botteghe vendevano frutta e carne e
attorno alle mura c'era il bosco fino al fossato.

Il **podere** e' un rettangolo arato con un canale d'acqua nel mezzo, e la
sua forma non e' una scelta estetica: in Minecraft la terra arata resta
bagnata solo entro quattro blocchi dall'acqua, altrimenti si secca e torna
terra. Un campo largo nove con il canale in mezzo e' bagnato tutto, uno largo
undici no. La forma DISCENDE dalla regola, come i ponti discendevano dalla
quota della strada. I poderi si spianano, come i lotti delle case.

Il **frutteto** e' a filari su un reticolo con un po' di disordine, chiuso
dal recinto. Il melo in Minecraft non esiste: e' la quercia, che le mele le
fa cadere davvero; il ciliegio invece c'e' dalla 1.20. Passo cinque e non
quattro, perche' la chioma di una latifoglia e' larga cinque blocchi e a
quattro di distanza i filari si saldano in un bosco unico.

Lungo i recinti, **siepi di bacche**.

Due difetti, trovati guardando:

* la coltura la sceglieva la singola cella e veniva fuori una scacchiera di
  grano, carote e patate dentro lo stesso campo. La coltura appartiene al
  podere, quindi ora sta nel codice della mappa - un valore per coltura -
  perche' il chunk, quando scrive i blocchi, non sa di quale podere sia la
  cella che ha davanti;
* le strade tagliavano dritto per i poderi. Nessuno lo aveva detto all'A*, e
  i campi sono la cosa piu' piana che ci sia: li avevamo appena spianati noi.
  Ora costano caro nella griglia di costo - caro, non vietato: una strada che
  deve passare di li' passa, e in campagna succede.

Su Arda: 26 poderi, 4 frutteti, 1.872 celle arate, 36 alberi da frutto, 350
cespugli di bacche, e zero celle di campo attraversate da una strada.

## Struttura

```
genworld/
  scale.py        modello delle scale, budget verticale, landmark
  fidelity.py     stimatore: una metrica per dimensione, selezione, avvisi
  presets.py      territori, citta' e monumenti di esempio
  report.py       rendering del pannello
  cli.py          riga di comando
  livello_dat.py  costruzione e verifica del level.dat
  mondo.py        scrittura Anvil per chunk in streaming
  anteprima.py    renderer top-down PNG del mondo generato
  rumore.py       fBm, dettaglio modulato dalla pendenza, quantizzazione ditherata
  mappa.py        import di mappe disegnate: classi, rugosita', quote dedotte
  normalizza.py   Lab, ritaglio cornice, bilanciamento, famiglia di mappa
  classi_auto.py  classificazione adattiva relativa (k-means in Lab)
  classi_guidate.py  classificazione da campioni, classe DECORO
  profilo.py      profilo di lettura: rettangolo, campioni, maschera dipinta
  erosione.py     erosione idraulica a gocce, vettoriale
  fiumi.py        deflusso D8, accumulo, livellamento dei corsi
  vegetazione.py  semina a griglia sfalsata, alberi per specie
  edifici.py      generatore parametrico, modificatore palafitta
  insediamenti.py scelta dei siti, lotti, terrazzamento
  strade.py       griglia di costo, A*, sede stradale, ponti
  vulcani.py      coni, valloni, cratere, lago di lava, colate
  citta.py        pianta organica, isolati, lotti sul fronte strada, mura
  entita.py       abitanti e scrittura a mano dei file entities/*.mca
  agricoltura.py  poderi, canali, recinti, frutteti a filari
  biomi.py        clima: freddo per quota e latitudine, tavolozza dei biomi
  motore.py       la pipeline: analizza, pianifica, scrivi
  gui.py          la finestra (PySide6)
tests/            196 test
esempi/
  spike_piatto.py generazione di prova 256x256 + anteprima
  genera_mappa.py riga di comando sopra il motore, a lotti
  prova_vulcani.py confronto visivo delle rugosita' del cono
  prova_citta.py  disegna la pianta urbana per guardarla
  scatto_gui.py   fotografa la finestra senza schermo
input/
  mappa_arda.png  mappa fantasy di prova
GenWorld.bat      avvio con doppio clic (crea l'ambiente al primo giro)
crea_exe.bat      impacchetta un .exe con PyInstaller (sperimentale)
avvia_gui.py      punto di ingresso per PyInstaller
```

## Prossimi passi

1. **Heightmap in scala di grigi** — oggi una mappa senza colore viene
   segnalata come non classificabile, quando invece e' il caso piu' semplice
   di tutti: i grigi *sono* le quote.
2. **Il Calibratore dentro la finestra** — rettangolo e contagocce vivono
   ancora in una pagina web a parte, e il profilo si passa come file.
3. **Operatori di generalizzazione** — semplificazione, selezione,
   tipificazione, e infine lo spostamento force-directed (il modulo piu'
   complesso del progetto).

La documentazione di progetto (architettura, altimetria e strutture, scale e
fedelta') sta nei doc del progetto Claude "Generatore di Mondi".
