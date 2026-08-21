class DataSet:

    def __init__(
        self,
        encoded_text,
        length,
        stride=1
    ):

        if not isinstance(stride, int) or stride < 1:
            stride = 1

        self.inputs = []
        self.outputs = []

        for i in range(
            0,
            len(encoded_text) - length - 1,
            stride
        ):

            self.inputs.append(
                encoded_text[i:i + length]
            )

            self.outputs.append(
                encoded_text[i + 1:i + length + 1]
            )


    def length(self):
        return len(self.inputs)


    def get(self, idx):
        return (
            self.inputs[idx],
            self.outputs[idx]
        )


class DataLoader:

    def __init__(
        self,
        raw_data,
        tokenizer,
        batch_size,
        length,
        stride=16
    ):

        if not isinstance(stride, int) or stride < 1:
            stride = 1

        if not isinstance(batch_size, int) or batch_size < 1:
            batch_size = 1

        if not isinstance(length, int) or length < 4:
            length = 4

        # Use the tokenizer passed from notebook
        encoded_text = tokenizer.encode(
            raw_data
        )

        self.dataset = DataSet(
            encoded_text,
            length,
            stride
        )

        self.batch_size = batch_size
        self.idx = 0


    def get_batch(self):

        inputs = []
        outputs = []

        for _ in range(self.batch_size):

            if self.idx >= self.dataset.length():
                break

            x, y = self.dataset.get(
                self.idx
            )

            inputs.append(x)
            outputs.append(y)

            self.idx += 1

        if len(inputs) < self.batch_size:
            return None, None

        return inputs, outputs


    def reset(self):
        self.idx = 0