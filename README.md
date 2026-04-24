# Green Rule RAG

Sistema per la generazione di regole trigger-action sostenibili per smart home, guidato da knowledge base e tecniche di Retrieval-Augmented Generation (RAG).

## Overview

Questo progetto esplora l’uso di tecniche di Retrieval-Augmented Generation (RAG) per supportare la generazione di regole trigger-action più sostenibili nel contesto delle smart home.

L’idea generale è partire da una regola espressa in linguaggio naturale, recuperare conoscenza rilevante da una knowledge base di dominio e usare tale contesto per guidare la generazione di una regola che migliori la sostenibilità energetica senza compromettere inutilmente il comportamento desiderato.

## Project Scope

Il progetto si concentra su un sistema capace di:

- analizzare regole trigger-action espresse in forma testuale;
- rappresentare gli elementi principali della regola in modo più strutturato;
- utilizzare una knowledge base come supporto alla generazione;
- recuperare informazione rilevante rispetto a una regola data;
- generare una regola sostenibile guidata dalla conoscenza recuperata;
- fornire un output comprensibile e coerente con l’intento iniziale;
- utilizzare eco-metriche per valutare la bontà delle regole generate.

## Goal

L’obiettivo del progetto è progettare e sviluppare un workflow che combini retrieval, knowledge base e generazione per produrre regole smart home orientate a una migliore sostenibilità.

Il sistema dovrà quindi collegare la descrizione di una regola alla conoscenza disponibile, usare il contesto recuperato per guidare la generazione della regola proposta e valutare il risultato tramite eco-metriche definite nel progetto.

## Expected Output

Per ogni regola in input, il sistema dovrebbe produrre almeno:

- la regola originale;
- la regola sostenibile generata;
- una breve descrizione delle modifiche introdotte;
- opzionalmente, una valutazione sintetica della regola generata basata su eco-metriche.

## Evaluation

La valutazione del progetto potrà considerare in modo generale:

- coerenza della regola generata rispetto all’intento originale;
- qualità della trasformazione proposta;
- contributo della regola al miglioramento della sostenibilità;
- pertinenza della conoscenza recuperata rispetto all’input;
- capacità delle eco-metriche di valutare la bontà della regola prodotta.