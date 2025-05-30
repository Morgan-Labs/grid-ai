import { useMemo, useEffect } from "react";
import {
  Box,
  BoxProps,
  Button,
  Divider,
  Group,
  TextInput,
  Text,
  Menu,
  ActionIcon,
  Select,
  NumberInput,
  TagsInput,
  Tooltip,
  Alert
} from "@mantine/core";
import {
  IconArrowAutofitHeight,
  IconDeviceFloppy,
  IconEyeOff,
  IconPlus,
  IconRefresh,
  IconTrash,
  IconRobot,
  IconBrandOpenai,
  IconBrandAws,
  IconBrandGoogle,
} from "@tabler/icons-react";
import { cloneDeep, isEmpty, isEqual, isString } from "lodash-es";
import { KtColumnQuestion } from "./kt-column-question";
import { generateOptions, typeOptions, llmOptions } from "./kt-column-settings.utils";
import {
  AnswerTableColumn,
  AnswerTableRule,
  defaultRules,
  isArrayType,
  ruleInfo,
  ruleOptions
} from "@config/store";
import { Empty, Info, MenuButton } from "@components";
import { useDerivedState } from "@hooks";
import { plur, where } from "@utils/functions";

interface Props extends BoxProps {
  value: AnswerTableColumn;
  onChange: (value: AnswerTableColumn, run?: boolean) => void;
  onRerun: () => void;
  onUnwind: () => void;
  onHide: () => void;
  onDelete: () => void;
}

export function KtColumnSettings({
  value,
  onChange,
  onRerun,
  onUnwind,
  onHide,
  onDelete,
  ...props
}: Props) {
  const [state, handlers] = useDerivedState(value, isEqual);

  useEffect(() => {
    if (handlers.dirty) {
      const timeoutId = setTimeout(() => {
        onChange(state);
      }, 2000); // 2 second delay before auto-saving
      return () => clearTimeout(timeoutId);
    }
  }, [state, handlers.dirty, onChange]);

  const typeOption = useMemo(
    () => typeOptions.find(option => option.value === state.type),
    [state.type]
  );
  const TypeIcon = typeOption?.icon;

  const handleSet = (value: Partial<AnswerTableColumn>) => {
    handlers.set(prev => ({ ...prev, ...value }));
  };

  const handleAddRule = () => {
    handleSet({
      rules: [...state.rules, { type: "must_return", options: [] }]
    });
  };

  const handleRuleTypeChange = (
    rule: AnswerTableRule,
    type: AnswerTableRule["type"]
  ) => {
    handleSet({
      rules: where(
        state.rules,
        r => r === rule,
        () => cloneDeep(defaultRules[type])
      )
    });
  };

  const handleRuleChange = (
    rule: AnswerTableRule,
    change: Pick<AnswerTableRule, "length" | "options">
  ) => {
    handleSet({
      rules: where(state.rules, r => r === rule, change)
    });
  };

  const handleDeleteRule = (rule: AnswerTableRule) => {
    handleSet({
      rules: state.rules.filter(r => r !== rule)
    });
  };

  const handleClipboardEvent = (e: React.ClipboardEvent) => {
    // Only prevent the event from bubbling up to parent components
    e.stopPropagation();
  };

  const rulesMenu = (
    <Box 
      w={430} 
    >
      {isEmpty(state.rules) ? (
        <div>
          <Empty message="No rules are applied" />
        </div>
      ) : (
        state.rules.map((rule, index) => (
          <Group 
            key={index} 
            mb="xs" 
            gap="xs" 
            wrap="nowrap"
          >
            <Select
              variant="unstyled"
              data={ruleOptions}
              comboboxProps={{ 
                withinPortal: false
              }}
              value={rule.type}
              onChange={type => {
                if (type) {
                  handleRuleTypeChange(rule, type as AnswerTableRule["type"]);
                }
              }}
            />
            {rule.type === "max_length" ? (
              <NumberInput
                w={150}
                min={1}
                decimalScale={0}
                placeholder="Max length"
                value={rule.length ?? 1}
                onChange={length =>
                  handleRuleChange(rule, {
                    length: isString(length) ? 1 : length
                  })
                }
              />
            ) : (
              <TagsInput
                w={210}
                placeholder="Values"
                value={rule.options ?? []}
                onChange={options => handleRuleChange(rule, { options })}
              />
            )}
            <div>
              <Info>
                {ruleInfo[rule.type]}
              </Info>
            </div>
            <ActionIcon 
              color="red" 
              onClick={() => handleDeleteRule(rule)}
            >
              <IconTrash />
            </ActionIcon>
          </Group>
        ))
      )}
      <Alert 
        mt="xs" 
      >
        Global rules may also apply to this column.
      </Alert>
      <Divider 
        mt="xs" 
      />
      <Group 
        mt="xs"
      >
        <Button 
          flex={1} 
          leftSection={<IconPlus />} 
          onClick={handleAddRule}
        >
          Add rule
        </Button>
        <Button
          flex={1}
          color="red"
          variant="light"
          leftSection={<IconTrash />}
          onClick={() => handleSet({ rules: [] })}
        >
          Clear
        </Button>
      </Group>
    </Box>
  );

  return (
    <Box {...props}
      onCopyCapture={handleClipboardEvent}
      onCutCapture={handleClipboardEvent}
      onPasteCapture={handleClipboardEvent}
    >
      <Group gap="xs" wrap="nowrap" justify="space-between">
        <TextInput
          autoFocus
          fw={500}
          size="md"
          variant="unstyled"
          label="Column Name"
          placeholder=""
          defaultValue={state.entityType}
          onChange={e => handleSet({ entityType: e.target.value })}
          required
        />
        <Group gap="xs" wrap="nowrap">
          <Tooltip label="Rerun column">
            <ActionIcon color="green" onClick={onRerun}>
              <IconRefresh />
            </ActionIcon>
          </Tooltip>
          {isArrayType(state.type) && (
            <Tooltip label="Split cells into rows">
              <ActionIcon onClick={onUnwind}>
                <IconArrowAutofitHeight />
              </ActionIcon>
            </Tooltip>
          )}
          <Tooltip label="Hide column">
            <ActionIcon onClick={onHide}>
              <IconEyeOff />
            </ActionIcon>
          </Tooltip>
          <Tooltip label="Delete column">
            <ActionIcon color="red" onClick={onDelete}>
              <IconTrash />
            </ActionIcon>
          </Tooltip>
        </Group>
      </Group>
      <Divider mt="xs" mx="calc(var(--mantine-spacing-sm) * -1)" />
      <MenuButton
        mt="xs"
        label="Type"
        rightSection={
          <Group c="dimmed">
            {TypeIcon && <TypeIcon size={16} />}
            <Text>{typeOption?.label}</Text>
          </Group>
        }
        menu={typeOptions.map(({ value: type, label, icon: Icon }) => (
          <Menu.Item
            key={type}
            onClick={() => handleSet({ type })}
            leftSection={<Icon />}
          >
            {label}
          </Menu.Item>
        ))}
      />
      <MenuButton
        mt="xs"
        label="Generate answers"
        rightSection={
          <Text c="dimmed">{state.generate ? "Enabled" : "Disabled"}</Text>
        }
        menu={generateOptions.map(({ value: generate, label, icon: Icon }) => (
          <Menu.Item
            key={String(generate)}
            onClick={() => handleSet({ generate })}
            leftSection={<Icon />}
          >
            {label}
          </Menu.Item>
        ))}
      />
      {state.generate && (
        <>
          <MenuButton
            mt="xs"
            label="LLM Model"
            rightSection={
              <Group c="dimmed">
                {state.llmModel && state.llmModel.includes("claude") && <IconBrandAws size={16} />}
                {state.llmModel && state.llmModel.includes("gemini") && <IconBrandGoogle size={16} />}
                {state.llmModel && (state.llmModel.includes("gpt")) && <IconBrandOpenai size={16} />}
                {!state.llmModel && <IconRobot size={16} />}
                <Text>
                  {state.llmModel || "Default"}
                </Text>
              </Group>
            }
            menu={
              <Box w={300} p="xs">
                {/* OpenAI Models */}
                <Box mb="md">
                  <Group mb="xs">
                    <IconBrandOpenai size={18} />
                    <Text fw={500}>OpenAI</Text>
                  </Group>
                  <Group gap="xs">
                    {llmOptions
                      .filter(model => model.provider === "openai")
                      .map(model => (
                        <Button
                          key={model.model}
                          size="xs"
                          variant={state.llmModel === model.model ? "filled" : "light"}
                          onClick={() => handleSet({
                            llmModel: model.model
                          })}
                        >
                          {model.label}
                        </Button>
                      ))
                    }
                  </Group>
                </Box>

                {/* Bedrock Anthropic Models */}
                <Box mb="md">
                  <Group mb="xs">
                    <IconBrandAws size={18} />
                    <Text fw={500}>AWS Bedrock Anthropic</Text>
                  </Group>
                  <Group gap="xs">
                    {llmOptions
                      .filter(model => model.provider === "bedrock")
                      .map(model => (
                        <Button
                          key={model.model}
                          size="xs"
                          variant={state.llmModel === model.model ? "filled" : "light"}
                          onClick={() => handleSet({
                            llmModel: model.model
                          })}
                        >
                          {model.label}
                        </Button>
                      ))
                    }
                  </Group>
                </Box>
                
                {/* Google Models */}
                <Box mb="md">
                  <Group mb="xs">
                    <IconBrandGoogle size={18} />
                    <Text fw={500}>Google</Text>
                  </Group>
                  <Group gap="xs">
                    {llmOptions
                      .filter(model => model.provider === "gemini")
                      .map(model => (
                        <Button
                          key={model.model}
                          size="xs"
                          variant={state.llmModel === model.model ? "filled" : "light"}
                          onClick={() => handleSet({
                            llmModel: model.model
                          })}
                        >
                          {model.label}
                        </Button>
                      ))
                    }
                  </Group>
                </Box>
                
                <Divider my="xs" />
                <Button
                  fullWidth
                  variant="light"
                  onClick={() => handleSet({
                    llmModel: undefined
                  })}
                >
                  Use Default Model
                </Button>
              </Box>
            }
          />
          <MenuButton
            mt="xs"
            label="Rules"
            dropdownProps={{ p: "sm" }}
            menu={rulesMenu}
            rightSection={
              <Text c="dimmed">
                {state.rules.length} {plur("rule", state.rules)}
              </Text>
            }
          />
          <KtColumnQuestion
            mt="xs"
            required
            label="Question"
            placeholder="Enter question (use '@' to reference other columns)"
            defaultValue={state.query}
            onChange={query => handleSet({ query })}
          />
        </>
      )}
      <Group mt="xs" wrap="nowrap">
        <Button
          flex={1}
          color="blue"
          variant="light"
          onClick={() => onChange(state)}
          disabled={!handlers.dirty}
          leftSection={<IconDeviceFloppy />}
        >
          Save
        </Button>
        {state.generate && (
          <Button
            flex={1}
            color="blue"
            variant="light"
            onClick={() => onChange(state, true)}
            disabled={!handlers.dirty}
            leftSection={<IconRefresh />}
          >
            Save and run
          </Button>
        )}{" "}
      </Group>
    </Box>
  );
}
