import React, { useState, useEffect } from "react";
import * as tf from "@tensorflow/tfjs";

const App = () => {
  const [data, setData] = useState([]);
  const [labels, setLabels] = useState([]);
  const [nextPrediction, setNextPrediction] = useState(null);
  const [inputValue, setInputValue] = useState("");

  const addData = () => {
    if (!inputValue) return;

    const newValues = inputValue.split(",").map(Number).filter(Boolean);
    let updatedData = [...data, ...newValues];
    let updatedLabels = [
      ...labels,
      ...newValues.map((sum) => (sum >= 3 && sum <= 10 ? 0 : 1)),
    ];

    while (updatedData.length > 100) {
      updatedData.shift();
      updatedLabels.shift();
    }

    setData(updatedData);
    setLabels(updatedLabels);
    setInputValue("");
  };

  useEffect(() => {
    if (data.length >= 100 && labels.length >= 100) {
      const model = tf.sequential();
      model.add(
        tf.layers.dense({ units: 10, activation: "relu", inputShape: [1] })
      );
      model.add(tf.layers.dense({ units: 1, activation: "sigmoid" }));

      model.compile({
        optimizer: tf.train.adam(),
        loss: "binaryCrossentropy",
        metrics: ["accuracy"],
      });

      const train = async () => {
        const xs = tf.tensor2d(data.map((d) => [d]));
        const ys = tf.tensor2d(labels.map((l) => [l]));
        await model.fit(xs, ys, { epochs: 100 });

        // Predict the next result
        const lastDataPoint = tf.tensor2d([data.slice(-1)], [1, 1]);
        const prediction = await model.predict(lastDataPoint).array();
        setNextPrediction(prediction[0][0]);
      };

      train();
    }
  }, [data, labels]);

  return (
    <div className="container">
      <div>
        <h1>Tính xác suất tài xỉu</h1>
        <input
          type="text"
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          placeholder="Nhập kết quả, ví dụ: 5,12,3"
        />
        <button onClick={addData}>Thêm dữ liệu</button>

        {nextPrediction !== null && (
          <div>
            <h2>Kết quả dự đoán ván tiếp theo:</h2>
            <p>
              Xác suất tài:{" "}
              <strong style={{ color: nextPrediction > 0.5 ? "red" : "black" }}>
                {(nextPrediction * 100).toFixed(2)}%
              </strong>
            </p>
            <p>
              Xác suất xỉu:{" "}
              <strong
                style={{ color: nextPrediction <= 0.5 ? "red" : "black" }}
              >
                {((1 - nextPrediction) * 100).toFixed(2)}%
              </strong>
            </p>
          </div>
        )}
      </div>
      <p>Tài: màu đỏ/ Xỉu: màu xanh</p>
      <div className="table">
        
        {data.map((foo, index) => {
          const isLastItem = index === data.length - 1;
          let additionalClass = "";

          if (foo <= 10) {
            additionalClass = "xiu";
          } else if (foo > 10) {
            additionalClass = "tai";
          }

          const finalClass = `${
            isLastItem ? "last-item" : ""
          } ${additionalClass}`;

          return <p className={finalClass.trim()}>{foo}</p>;
        })}
      </div>
    </div>
  );
};

export default App;
