import React from 'react';
import { Text, View, FlatList, StyleSheet, TouchableOpacity } from 'react-native';

export default class Question extends React.Component {

  constructor(props) {
    super(props);
    this.state = {
      showTrueColor: props.showTrueColor
    };
  }


  // componentWillReceiveProps(props) {
  //
  //   console.log(props);
  //   const { item, showTrueColor } = this.props;
  //
  //   console.log("Current color is ", showTrueColor);
  //   console.log("New color is ", props.showTrueColor);
  //   if(showTrueColor !== props.showTrueColor) {
  //     this.setState({color: props.showTrueColor})
  //   }
  // }

  state = {
    showTrueColor: false
  }

  // getColor = ({item}) => {
  //
  //   console.log("Trying to get get color");
  //   console.log("Current state is", this.props.showTrueColor);
  //
  //   if(this.props.showTrueColor) {
  //     return item.color;
  //   }
  //   return 'yellow';
  // }

  render() {

    console.log("Render", this.props.item);
    if(this.props.showTrueColor) {

      return (
        <View style={styles.questionView}>
          <Text style={styles.question}>{this.props.item.question}</Text>
          <FlatList
            showTrueColor={this.state.showTrueColor}
            style={styles.list}
            data={this.props.item.choices}
            renderItem={({item}) =>
              <Text style={[styles.answer, {backgroundColor: item.color}]}>{item.text}</Text>
            }
          />
        </View>
      );
    } else {
      console.log("Hello!");
      return (
        <View style={styles.questionView}>
          <Text style={styles.question}>{this.props.item.question}</Text>
          <FlatList
            style={styles.list}
            data={this.props.item.choices}
            renderItem={({item}) =>
              <TouchableOpacity onPress={this.props.checkAnswer}>
                <Text style={[styles.answer, {backgroundColor: 'yellow'}]}>{item.text}</Text>
              </TouchableOpacity>
            }
          />
        </View>
      );
    }
  }
}

const styles = StyleSheet.create({
  questionView: {
    flexDirection: 'column',
    flex: 1,
  },
  question: {
    textAlign: 'center',
    fontSize: 25,
    flex: 1,
  },
  answer: {
    textAlign: 'center',
    fontSize: 25,
    margin: 10,
    backgroundColor: 'red',
  },
  list: {
    flex: 10,
  }
})
