import 'cheerio';
import { CheerioWebBaseLoader } from '@langchain/community/document_loaders/web/cheerio';
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import { MemoryVectorStore } from 'langchain/vectorstores/memory';
import { OpenAIEmbeddings, ChatOpenAI } from '@langchain/openai';
import { pull } from 'langchain/hub';
import { ChatPromptTemplate, PromptTemplate } from '@langchain/core/prompts';
import { StringOutputParser } from '@langchain/core/output_parsers';
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { formatDocumentsAsString } from "langchain/util/document";
import { RunnableSequence, RunnablePassthrough } from '@langchain/core/runnables';

const loader = new CheerioWebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/");

const docs = await loader.load();

const textSplitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
    chunkOverlap: 200,
});
const splits = await textSplitter.splitDocuments(docs);
const vectorStore = await MemoryVectorStore.fromDocuments(splits, new OpenAIEmbeddings());

const retriever = vectorStore.asRetriever({
    k: 6,
    searchType: 'similarity',
});

const customTemplate = `Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use three sentences maximum and keep the answer as concise as possible.
Always say "thanks for asking!" at the end of the answer.

{context}

Question: {question}

Helpful Answer:`;

const prompt = PromptTemplate.fromTemplate(customTemplate);

// /**
//  * @type {ChatPromptTemplate}
//  */
// const prompt = await pull('rlm/rag-prompt');

const llm = new ChatOpenAI({
    model: 'gpt-4o-mini',
    temperature: 0,
});

// const ragChain = await createStuffDocumentsChain({
//     llm,
//     prompt,
//     outputParser: new StringOutputParser(),
// });

const question = "What are the approaches to task decomposition?";

// const retrievedDocs = await retriever.invoke(question);

const ragChain = RunnableSequence.from([
    {
        context: retriever.pipe(formatDocumentsAsString),
        question: new RunnablePassthrough(),
    },
    prompt,
    llm,
    new StringOutputParser(),
]);

// const ragState = await ragChain.invoke(question);

// console.log(ragState);

for await (const chunk of await ragChain.stream(question)) {
    process.stdout.write(chunk);
}
console.log('');