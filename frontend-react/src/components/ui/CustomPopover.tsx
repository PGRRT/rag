import colorPalette from "@/constants/colorPalette";
import { styles } from "@/constants/styles";
import { Popover } from "@mantine/core";

const CustomPopover = ({
  trigger,
  content,
  open,
  setOpen,
  width,
  position,
}: {
  trigger: React.ReactNode;
  content: React.ReactNode;
  open: boolean;
  setOpen: (open: boolean) => void;
  width?: number | string;
  position?:
    | "top"
    | "bottom"
    | "left"
    | "right"
    | "top-start"
    | "top-end"
    | "bottom-start"
    | "bottom-end"
    | "left-start"
    | "left-end"
    | "right-start"
    | "right-end";
}) => {
  return (
    <>
      <Popover
        opened={open}
        onChange={setOpen}
        width={width ?? 200}
        position={position ?? "bottom-start"}
        styles={{
          dropdown: {
            borderRadius: styles.borderRadius.small,
            background: colorPalette.backgroundBright,
            border: "unset",
          },
        }}
      >
        <Popover.Target>{trigger}</Popover.Target>

        <Popover.Dropdown>{content}</Popover.Dropdown>
      </Popover>
    </>
  );
};

export default CustomPopover;
