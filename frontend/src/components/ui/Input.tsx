import type { ChangeEvent, InputHTMLAttributes } from "react";
import ContentWrapper from "@/components/ui/ContentWrapper";
import { typography } from "@/constants/typography";
import { colorPalette } from "@/constants/colorPalette";
import { css } from "@emotion/react";

interface InputProps
  extends Omit<InputHTMLAttributes<HTMLInputElement>, "onChange"> {
  value?: string;
  onChange?: (e: ChangeEvent<HTMLInputElement>) => void;
  label?: string;
  error?: string;
  description?: string;
  className?: string;
}

const Input: React.FC<InputProps> = ({
  value,
  onChange,
  label,
  error,
  className = "",
  description,
  ...props
}) => {
  return (
    <ContentWrapper gap="10px" direction="column">
      {label && (
        <label
          htmlFor={props.name}
          className="text-sm font-medium text-gray-700"
        >
          {label}
        </label>
      )}
      <input
        aria-invalid={error ? "true" : "false"}
        id={props.name}
        value={value}
        onChange={onChange}
        {...props}
        className={`
          px-4 py-2 border rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500
          disabled:bg-gray-100 disabled:cursor-not-allowed
          border-gray-300 ${error ? "border-red-500" : ""}
          ${className}
        `}
      />
      {description && (
        <p
          css={[
            typography.textS,
            css`
              color: #6f6f6f;
            `,
          ]}
        >
          {description}
        </p>
      )}
      {error && <p className="text-red-500 text-sm">{error}</p>}
    </ContentWrapper>
  );
};

export default Input;
