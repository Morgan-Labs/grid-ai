import { ReactNode } from "react";
import {
  Box,
  BoxProps,
  Group,
  Input,
  Menu,
  MenuDropdownProps,
  MenuProps
} from "@mantine/core";
import { IconChevronRight } from "@tabler/icons-react";
import { cn } from "@utils/functions";
import classes from "./index.module.css";

interface Props extends BoxProps {
  disabled?: boolean;
  label: ReactNode;
  rightSection?: ReactNode;
  menu?: ReactNode;
  menuProps?: MenuProps;
  dropdownProps?: MenuDropdownProps;
}

export function MenuButton({
  label,
  rightSection,
  disabled,
  menu,
  menuProps,
  dropdownProps,
  ...props
}: Props) {
  return (
    <Menu
      position="right-start"
      offset={2}
      withinPortal={true}
      disabled={disabled}
      closeOnItemClick={false}
      {...menuProps}
    >
      <Menu.Target>
        <Box 
          {...props} 
          className={cn(classes.menuButton, props.className)}
          onClick={(e) => {
            e.preventDefault();
            e.stopPropagation();
          }}
          data-mantine-stop-propagation="true"
        >
          <Input.Label>{label}</Input.Label>
          <Group>
            <Box>{rightSection}</Box>
            <IconChevronRight size={16} />
          </Group>
        </Box>
      </Menu.Target>
      <Menu.Dropdown 
        {...dropdownProps} 
        onPointerDown={(e) => e.stopPropagation()}
        onKeyDown={(e) => e.stopPropagation()}
        onClick={(e) => e.stopPropagation()}
        data-mantine-stop-propagation="true"
      >
        <div 
          onClick={(e) => e.stopPropagation()}
          onMouseDown={(e) => e.stopPropagation()}
          data-mantine-stop-propagation="true"
        >
          {menu}
        </div>
      </Menu.Dropdown>
    </Menu>
  );
}
