import { ReactNode } from "react";
import {
  IconAccessPoint,
  IconAccessPointOff,
  IconAlignJustified,
  IconCheckbox,
  IconHash,
  TablerIcon,
} from "@tabler/icons-react";
import { AnswerTableColumn } from "@config/store";

export const typeOptions: {
  value: AnswerTableColumn["type"];
  label: ReactNode;
  icon: TablerIcon;
}[] = [
  { value: "str", label: "Text", icon: IconAlignJustified },
  { value: "str_array", label: "List of text", icon: IconAlignJustified },
  { value: "int", label: "Number", icon: IconHash },
  { value: "int_array", label: "List of numbers", icon: IconHash },
  { value: "bool", label: "True / False", icon: IconCheckbox }
];

export const generateOptions: {
  value: boolean;
  label: ReactNode;
  icon: TablerIcon;
}[] = [
  { value: true, label: "Enabled", icon: IconAccessPoint },
  { value: false, label: "Disabled", icon: IconAccessPointOff }
];

// LLM options - simplified to just include the model name since we detect provider server-side
export interface LLMOption {
  provider: string;
  model: string;
  label: string;
}

export const llmOptions: LLMOption[] = [
  // OpenAI models
  { provider: "openai", model: "gpt-4o", label: "GPT-4o" },
  { provider: "openai", model: "gpt-4o-mini", label: "GPT-4o Mini" },
  { provider: "openai", model: "gpt-4-turbo", label: "GPT-4 Turbo" },
  
  // Anthropic models
  { provider: "anthropic", model: "claude-3-7-sonnet-20250219", label: "Claude 3.7 Sonnet" },
  { provider: "anthropic", model: "claude-3-5-sonnet-20241022", label: "Claude 3.5 Sonnet" },
  { provider: "anthropic", model: "claude-3-opus-20240229", label: "Claude 3 Opus" },
  
  // Google models
  { provider: "gemini", model: "gemini-2.5-pro-exp-03-25", label: "Gemini 2.5 Pro Exp" },
  { provider: "gemini", model: "gemini-2.0-flash", label: "Gemini 2.0 Flash" },
  { provider: "gemini", model: "gemini-2.0-flash-lite", label: "Gemini 2.0 Flash Lite" },
  { provider: "gemini", model: "gemini-1.5-flash", label: "Gemini 1.5 Flash" }
];
