import { Button, TextInput, Select, Card, Paper, Tooltip } from "@mantine/core";
import { Send } from "lucide-react";

/**
 * Example component showing how to use Mantine components with your custom theme.
 * All components will automatically use your color palette and spacing.
 */
export default function MantineExamples() {
  return (
    <div
      style={{
        padding: "2rem",
        display: "flex",
        flexDirection: "column",
        gap: "1rem",
      }}
    >
      <h2>Mantine Components Examples</h2>

      {/* Buttons with different variants */}
      <div style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap" }}>
        <Button variant="filled">Filled Button</Button>
        <Button variant="outline">Outline Button</Button>
        <Button variant="light">Light Button</Button>
        <Button variant="subtle">Subtle Button</Button>
        <Button leftSection={<Send size={16} />}>With Icon</Button>
      </div>

      {/* Text Input */}
      <TextInput
        label="Email"
        placeholder="Enter your email"
        description="We'll never share your email"
      />

      {/* Select */}
      <Select
        label="Choose option"
        placeholder="Pick one"
        data={[
          { value: "1", label: "Option 1" },
          { value: "2", label: "Option 2" },
          { value: "3", label: "Option 3" },
        ]}
      />

      {/* Card */}
      <Card shadow="sm" padding="lg" radius="md" withBorder>
        <h3>Card Title</h3>
        <p>This card uses your custom theme colors automatically.</p>
        <Button variant="light" fullWidth mt="md">
          Action
        </Button>
      </Card>

      {/* Paper */}
      <Paper shadow="xs" p="md">
        <p>Paper component with custom background</p>
      </Paper>

      {/* Tooltip */}
      <Tooltip label="This is a tooltip">
        <Button>Hover me</Button>
      </Tooltip>
    </div>
  );
}
