# Green Rule RAG

A system for generating sustainable trigger-action rules for smart homes, guided by a knowledge base and Retrieval-Augmented Generation (RAG) techniques.

## Overview

This project explores the use of Retrieval-Augmented Generation (RAG) techniques to support the creation of more sustainable trigger-action rules in the smart home domain.

The general idea is to start from a rule expressed in natural language, retrieve relevant knowledge from a domain knowledge base, and use that context to guide the generation of a rule that improves energy sustainability without unnecessarily compromising the desired behavior.

## Project Scope

The project focuses on a system capable of:

- analyzing trigger-action rules expressed in text form;
- representing the main elements of a rule in a more structured way;
- using a knowledge base as support for generation;
- retrieving information relevant to a given rule;
- generating a sustainable rule guided by retrieved knowledge;
- producing an output that is understandable and consistent with the original intent;
- using eco-metrics to evaluate the quality of the generated rules.

## Goal

The goal of the project is to design and develop a workflow that combines retrieval, a knowledge base, and generation to produce smart home rules oriented toward better sustainability.

The system should therefore connect the description of a rule to the available knowledge, use the retrieved context to guide the generation of the proposed rule, and evaluate the result through eco-metrics defined within the project.

## Expected Output

For each input rule, the system should produce at least:

- the original rule;
- the generated sustainable rule;
- a short description of the modifications introduced;
- optionally, a concise evaluation of the generated rule based on eco-metrics.

## Evaluation

The evaluation of the project may generally consider:

- consistency of the generated rule with the original intent;
- quality of the proposed transformation;
- contribution of the rule to sustainability improvement;
- relevance of the retrieved knowledge with respect to the input;
- ability of eco-metrics to assess the quality of the produced rule.
