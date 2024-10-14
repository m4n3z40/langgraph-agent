import { TavilySearchResults } from "@langchain/community/tools/tavily_search";
import { ChatOpenAI } from "@langchain/openai";
import { MemorySaver, StateGraph, MessagesAnnotation } from "@langchain/langgraph";
import { HumanMessage } from "@langchain/core/messages";
import { ToolNode } from "@langchain/langgraph/prebuilt";

// Define the tools for the agent to use
const agentTools = [new TavilySearchResults({ maxResults: 3 })];
const toolsNode = new ToolNode(agentTools);
const agentModel = new ChatOpenAI({ 
    model: 'gpt-4o-mini',
    temperature: 0 
}).bindTools(agentTools);

/**
 * @param {typeof MessagesAnnotation.State} state
 * @returns string
 */
const shouldContinue = ({ messages }) => {
    const lastMessage = messages[messages.length - 1];

    if (lastMessage.additional_kwargs.tool_calls) {
        return 'tools';
    }

    return '__end__';
};

/**
 * @param {typeof MessagesAnnotation.State} state
 * @returns string
 */
const callModel = async (state) => {
    const response = await agentModel.invoke(state.messages);

    return { messages: [response] };
};

const workflow = new StateGraph(MessagesAnnotation)
    .addNode('agent', callModel)
    .addEdge('__start__', 'agent')
    .addNode('tools', toolsNode)
    .addEdge('tools', 'agent')
    .addConditionalEdges('agent', shouldContinue);

const app = workflow.compile();

const agentFinalState = await app.invoke(
    { messages: [new HumanMessage("como está o clima atualmente em san francisco?")] }
);

console.log(agentFinalState.messages[agentFinalState.messages.length - 1].content);

const agentNextState = await app.invoke(
    { messages: [...agentFinalState.messages, new HumanMessage("e nova iorque?")] },
);

console.log(agentNextState.messages[agentNextState.messages.length - 1].content);
