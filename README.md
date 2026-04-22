# Smart Home Green RAG

Sistema per la generazione di varianti sostenibili di regole trigger-action per smart home, con preservazione dell’intento utente e supporto tramite knowledge base.

## Overview

Questo progetto esplora l’uso di tecniche di Retrieval-Augmented Generation (RAG) per supportare la trasformazione di regole trigger-action in varianti più sostenibili nel contesto delle smart home.

L’idea generale è partire da una regola espressa in linguaggio naturale, recuperare conoscenza rilevante da una knowledge base di dominio e produrre una variante che migliori la sostenibilità energetica senza compromettere inutilmente il comportamento desiderato.

## Project Scope

Il progetto si concentra su un sistema capace di:

- analizzare regole trigger-action espresse in forma testuale;
- rappresentare gli elementi principali della regola in modo più strutturato;
- utilizzare una knowledge base come supporto alla generazione;
- recuperare informazione rilevante rispetto a una regola data;
- generare una o più varianti sostenibili della regola originale;
- utilizzare eco-metriche per analizzare e confrontare le varianti generate;
- fornire un output comprensibile e coerente con l’intento iniziale.

## Goal

L’obiettivo del progetto è progettare e sviluppare un workflow che combini retrieval, generazione e valutazione per proporre versioni alternative di regole smart home orientate a una migliore sostenibilità.

Il sistema dovrà quindi collegare la descrizione di una regola alla conoscenza disponibile, usare tale contesto per guidare la generazione della variante proposta e supportare la selezione della soluzione più “green” tramite eco-metriche definite nel progetto.

## Expected Output

Per ogni regola in input, il sistema dovrebbe produrre almeno:

- la regola originale;
- una o più varianti sostenibili generate;
- una variante selezionata come output finale;
- una breve descrizione delle modifiche introdotte;
- opzionalmente, un punteggio o una valutazione sintetica associata alle varianti.

## Evaluation

La valutazione del progetto potrà considerare in modo generale:

- coerenza della variante rispetto all’intento originale della regola;
- qualità della trasformazione proposta;
- contributo della variante al miglioramento della sostenibilità;
- pertinenza della conoscenza recuperata rispetto all’input;
- capacità delle eco-metriche di distinguere tra varianti più o meno sostenibili.
