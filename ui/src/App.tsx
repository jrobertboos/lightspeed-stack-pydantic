import { useEffect, useMemo, useRef, useState } from 'react';
import { DropdownItem, DropdownList } from '@patternfly/react-core';
import {
  Chatbot,
  ChatbotContent,
  ChatbotDisplayMode,
  ChatbotFooter,
  ChatbotHeader,
  ChatbotHeaderActions,
  ChatbotHeaderMain,
  ChatbotHeaderNewChatButton,
  ChatbotHeaderSelectorDropdown,
  ChatbotHeaderTitle,
  ChatbotWelcomePrompt,
  Message,
  MessageBar,
  MessageBox,
} from '@patternfly/chatbot';

import { listModels, streamQuery } from './api/client';
import { ApiError, type ModelInfo } from './api/types';

interface DisplayMessage {
  id: string;
  role: 'user' | 'bot';
  content: string;
  timestamp: string;
  isLoading?: boolean;
  error?: string;
}

function timestamp(): string {
  return new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

const DEFAULT_MODEL_IDENTIFIER = 'openai/gpt-4o-mini';

function App() {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [modelsError, setModelsError] = useState<string | null>(null);
  const [selectedModel, setSelectedModel] = useState<ModelInfo | null>(null);

  const [conversationId, setConversationId] = useState<string | undefined>(undefined);
  const [messages, setMessages] = useState<DisplayMessage[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);

  const abortControllerRef = useRef<AbortController | null>(null);

  // Models come exclusively from `GET /v1/models`; the UI has no built-in list.
  useEffect(() => {
    listModels()
      .then((res) => {
        setModels(res.models);
        setSelectedModel(
          (current) =>
            current ??
            res.models.find((m) => m.identifier === DEFAULT_MODEL_IDENTIFIER) ??
            res.models[0] ??
            null,
        );
      })
      .catch((err: unknown) => {
        setModelsError(err instanceof ApiError ? err.message : 'Failed to load models');
      });
  }, []);

  const modelSelectorValue = useMemo(
    () => (selectedModel ? selectedModel.identifier : 'No models available'),
    [selectedModel],
  );

  function handleNewChat() {
    abortControllerRef.current?.abort();
    setIsStreaming(false);
    setConversationId(undefined);
    setMessages([]);
  }

  function handleStop() {
    abortControllerRef.current?.abort();
    setIsStreaming(false);
  }

  async function handleSendMessage(message: string | number) {
    const text = String(message).trim();
    if (!text || !selectedModel || isStreaming) {
      return;
    }

    const userMessage: DisplayMessage = {
      id: crypto.randomUUID(),
      role: 'user',
      content: text,
      timestamp: timestamp(),
    };
    const botMessageId = crypto.randomUUID();
    const botMessage: DisplayMessage = {
      id: botMessageId,
      role: 'bot',
      content: '',
      timestamp: timestamp(),
      isLoading: true,
    };
    setMessages((prev) => [...prev, userMessage, botMessage]);
    setIsStreaming(true);

    const controller = new AbortController();
    abortControllerRef.current = controller;

    function patchBotMessage(patch: Partial<DisplayMessage>) {
      setMessages((prev) => prev.map((m) => (m.id === botMessageId ? { ...m, ...patch } : m)));
    }

    try {
      for await (const event of streamQuery(
        {
          query: text,
          conversation_id: conversationId,
          provider: selectedModel.provider_name,
          model: selectedModel.model_name,
        },
        controller.signal,
      )) {
        switch (event.event) {
          case 'start':
            setConversationId(event.data.conversation_id);
            break;
          case 'token':
            setMessages((prev) =>
              prev.map((m) =>
                m.id === botMessageId ? { ...m, content: m.content + event.data.token, isLoading: true } : m,
              ),
            );
            break;
          case 'turn_complete':
            patchBotMessage({ content: event.data.token, isLoading: false });
            break;
          case 'end':
            patchBotMessage({ isLoading: false });
            break;
          case 'error':
            patchBotMessage({
              isLoading: false,
              content: event.data.response,
              error: event.data.cause,
            });
            break;
        }
      }
    } catch (err) {
      if (!(err instanceof DOMException && err.name === 'AbortError')) {
        patchBotMessage({
          isLoading: false,
          content: err instanceof ApiError ? err.message : 'Something went wrong while streaming the response.',
          error: err instanceof ApiError ? err.cause_ : String(err),
        });
      }
    } finally {
      setIsStreaming(false);
      abortControllerRef.current = null;
    }
  }

  return (
    <Chatbot displayMode={ChatbotDisplayMode.fullscreen}>
      <ChatbotHeader>
        <ChatbotHeaderMain>
          <ChatbotHeaderTitle>Lightspeed Stack</ChatbotHeaderTitle>
        </ChatbotHeaderMain>
        <ChatbotHeaderActions>
          <ChatbotHeaderSelectorDropdown
            value={modelSelectorValue}
            maxMenuHeight="40vh"
            isScrollable
            onSelect={(_event, value) => {
              const model = models.find((m) => m.identifier === value);
              if (model) setSelectedModel(model);
            }}
          >
            <DropdownList>
              {models.map((model) => (
                <DropdownItem key={model.identifier} value={model.identifier}>
                  {model.identifier}
                </DropdownItem>
              ))}
            </DropdownList>
          </ChatbotHeaderSelectorDropdown>
          <ChatbotHeaderNewChatButton onClick={handleNewChat} />
        </ChatbotHeaderActions>
      </ChatbotHeader>
      <ChatbotContent>
        <MessageBox>
          {modelsError && (
            <Message role="bot" content={`Unable to load models: ${modelsError}`} timestamp={timestamp()} />
          )}
          {!modelsError && messages.length === 0 && (
            <ChatbotWelcomePrompt
              title="Welcome to Lightspeed Stack"
              description="Ask a question to get started. Responses come straight from the configured provider/model."
            />
          )}
          {messages.map((m) => (
            <Message
              key={m.id}
              role={m.role}
              content={m.content}
              timestamp={m.timestamp}
              isLoading={m.isLoading}
              error={m.error ? { title: 'Error', children: m.error } : undefined}
            />
          ))}
        </MessageBox>
      </ChatbotContent>
      <ChatbotFooter>
        <MessageBar
          onSendMessage={handleSendMessage}
          hasStopButton={isStreaming}
          handleStopButton={handleStop}
          isSendButtonDisabled={!selectedModel || isStreaming}
          placeholder={selectedModel ? 'Send a message...' : 'Waiting for models to load...'}
        />
      </ChatbotFooter>
    </Chatbot>
  );
}

export default App;
