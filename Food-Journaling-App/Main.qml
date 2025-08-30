import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "UI-Items"

ApplicationWindow {
    id: window
    visible: true
    title: qsTr("Check Macros")

    // Fill available screen space
    width: Screen.width
    height: Screen.height

    // Define a scale factor based on screen size (for padding, font sizes, etc.)
    property real scale: Math.min(width, height) / 400

    ColumnLayout {
        anchors.centerIn: parent
        spacing: 20 * scale
        width: parent.width * 0.8   // make it proportional
        height: parent.height * 0.6

        TextField {
            id: inputField
            Layout.fillWidth: true
            placeholderText: qsTr("Enter food (e.g. 200g chicken breast)")
            font.pixelSize: 14 * scale
        }

        Button {
            Layout.fillWidth: true
            text: qsTr("Check")
            font.pixelSize: 14 * scale
            onClicked: appController.predict(inputField.text)
        }

        PredictionCard {
            Layout.fillWidth: true
            Layout.fillHeight: true
            calories: appController.calories
            protein: appController.protein
            fat: appController.fat
            carbs: appController.carbs
        }
    }
}
