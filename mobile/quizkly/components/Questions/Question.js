import React from 'react';
import { Text, View, FlatList, StyleSheet } from 'react-native';

export default class Question extends React.Component {

  constructor(props) {
    super(props);
    this.state = {
      showTrueColor: props.showTrueColor
    };
    console.log(this.state.showTrueColor);
    console.log("That's as I'm constructed!");
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

    console.log(this.props.showTrueColor);
    console.log("Was that good?");

    if(this.props.showTrueColor) {

      console.log("About to render real color!");
      console.log(this.props.item.choices);
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

      console.log("About to render yellow color!");
      return (
        <View style={styles.questionView}>
          <Text style={styles.question}>{this.props.item.question}</Text>
          <FlatList
            style={styles.list}
            data={this.props.item.choices}
            renderItem={({item}) =>
              <Text style={[styles.answer, {backgroundColor: 'yellow'}]}>{item.text}</Text>
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
  },
  question: {
    textAlign: 'center',
    fontSize: 20,
    flex: 1,
  },
  answer: {
    textAlign: 'center',
    fontSize: 15,
    margin: 10,
    backgroundColor: 'red',
  },
  list: {
    flex: 10,
  }
})
