# Case da template

I file `.nbt` di questa cartella sono case. GenWorld le legge tutte all'avvio
e, per ogni lotto che progetta, ne pesca una a caso fra quelle che ci stanno
dentro e che possono guardare la strada. Dove non ci sta nessun template si
torna al generatore parametrico, che nel lotto entra sempre perche' e' il
lotto a dargli le misure.

Aggiungere una casa vuol dire mettere un file qui. Toglierne una vuol dire
cancellarlo. Non c'e' nient'altro da configurare.

## Farne una nuova, dentro Minecraft

1. Apri un mondo in creativa e costruisci la casa.
2. Prendi un **blocco struttura**: `/give @p structure_block`.
3. Mettilo a un angolo della casa, in basso. Modalita' **SAVE**.
4. Dagli un nome (per esempio `mia_casa`) e regola `size` e `offset` finche'
   il riquadro bianco contiene tutta la casa, tetto e gronde comprese.
5. **Salva.** Minecraft scrive il file in
   `.minecraft/saves/<mondo>/generated/minecraft/structures/mia_casa.nbt`.
6. Copia quel file qui dentro. Fatto.

## Tre regole che contano

**Lo strato y=0 e' il pavimento.** GenWorld posa la struttura in modo che il
suo livello piu' basso finisca sulla quota del terreno spianato. Quindi il
riquadro deve cominciare dal pavimento, non da tre blocchi di terra sotto: se
salvi anche il terreno, la casa viene fuori sollevata di tre blocchi.

**La porta decide da che parte guarda la casa.** Il programma cerca una porta
nella struttura e ne legge l'orientamento, poi ruota la casa in modo che
guardi la strada. Una casa senza porta viene messa con una rotazione
qualunque - funziona, ma prima o poi ne trovi una che da' le spalle alla via.

**Piu' e' piccola, piu' viene usata.** I lotti di una pianta organica sono
quasi tutti piccoli: una casa di 7x7 entra quasi ovunque, una di 13x13 quasi
da nessuna parte. Se ne aggiungi una grande, aggiungine anche due piccole.

`structure_void` funziona come ci si aspetta: quelle celle non vengono
toccate, e servono per case di pianta non rettangolare.

## Quelle che ci sono adesso

Sono generate da `esempi/esporta_template.py`, che esporta le case
parametriche in questo formato. Ci sono perche' la cartella non sia vuota
appena scarichi il progetto, e soprattutto perche' sono un punto di partenza:
e' molto piu' facile sistemare una casa che disegnarne una da zero. Caricane
una con un blocco struttura in modalita' LOAD, sistemala, risalvala con lo
stesso nome.

Per rifarle:

    python esempi/esporta_template.py

## Se una casa non compare

Il caricamento salta i file rotti e stampa il motivo:

    template saltato: mia_casa.nbt (...)

Le cause solite sono un file che non e' NBT compresso (per esempio uno
`.schem` di WorldEdit, che e' un altro formato), oppure una struttura piu'
alta di ventiquattro blocchi, che viene scartata perche' non ci sta nel
budget verticale dei lotti.
