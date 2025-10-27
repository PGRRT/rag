import ContentWrapper from "@/components/ui/ContentWrapper";
import CustomPopover from "@/components/ui/CustomPopover";
import MantineExamples from "@/components/ui/MantineExamples";
import SelectInput from "@/components/ui/SelectInput";
import { Button } from "@mantine/core";
import { useState } from "react";

const ragOptions = [
  {
    label: "Classical Rag",
    value: "classical_rag",
  },
  {
    label: "Jazz Rag",
    value: "jazz_rag",
  },
];
const Navbar = () => {
  const [rag, setRag] = useState(ragOptions[0].value);

  return (
    <>
      <ContentWrapper justify="space-between">
        <SelectInput value={rag} onChange={setRag} options={ragOptions} />

        <ContentWrapper direction="row" gap="10px">
          <Button>Sign in</Button>
          <Button variant="outline">Sign up</Button>
        </ContentWrapper>
      </ContentWrapper>
      {/* <MantineExamples /> */}
    </>
  );
};

export default Navbar;
