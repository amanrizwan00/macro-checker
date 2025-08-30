import QtQuick
import QtQuick.Controls

Rectangle {
    property double calories: 0
    property double protein: 0
    property double fat: 0
    property double carbs: 0

    width: parent.width
    height: 150
    radius: 10
    color: "#f0f0f0"

    Column {
        anchors.centerIn: parent
        spacing: 8
        Text { text: "Calories: " + calories.toFixed(1) + " kcal" }
        Text { text: "Protein: "  + protein.toFixed(1) + " g" }
        Text { text: "Fat: "      + fat.toFixed(1) + " g" }
        Text { text: "Carbs: "    + carbs.toFixed(1) + " g" }
    }
}
