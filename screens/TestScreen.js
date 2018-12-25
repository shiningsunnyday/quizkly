import React from 'react';
import { ScrollView, FlatList, Text, StyleSheet, Button, View, SectionList } from 'react-native';
import Question from '../components/Questions/Question';
import { QuestionList } from '../components/Questions/QuestionList';

export default class TestScreen extends React.Component {

  static navigationOptions = {
    title: 'Test',
  };

  state = {
    params: this.props.navigation.state.params.params,
    curItemIndex: 0,
    curItem: this.props.navigation.state.params.params.list[0],
    showTrueColor: false,
  }

  changeQuestion() {

    if(this.state.curItemIndex < params.list.length - 1) {
      let newQuestion = params.list[this.state.curItemIndex + 1];
      console.log(newQuestion.question, "is the next question in line!");
      this.setState({
        curItem: newQuestion,
        curItemIndex: this.state.curItemIndex + 1,
      })
    } else {
      this.setState({
        curItemIndex: 0,
      })
    }
  }

  render() {

    return (
        // <FlatList style={styles.flatList} data={params.list}
        //   renderItem={({item, index, section}) =>
        //     <View style={styles.questionView} key={index}>
        //       <Text style={styles.questionText}>{item.question}</Text>
        //     </View>}
        // />
        <View style={styles.container}>
          <View style={styles.container}>
            <Question item={this.state.curItem} showTrueColor={this.state.showTrueColor} />
          </View>
          <Button title="Next Question!" onPress={this.changeQuestion.bind(this)} style={styles.questionButton}/>
        </View>
    );
  }
}

const styles = StyleSheet.create({
  question: {
    backgroundColor: 'red',
    flex: 1,
  },
  questionButton: {
    flex: 1,
    backgroundColor: 'yellow',
  },
  container: {
    flex: 1,
    flexDirection: 'column',
    justifyContent: 'center',
  },
  flatList: {
    flex: 1,
    flexDirection: 'column',
    backgroundColor: 'green',
  },
  flatListContent: {
    flexDirection: 'column',
    backgroundColor: 'yellow',
    justifyContent: 'center',
    alignItems: 'center',
  },
  questionText: {
    flex: 1,
    flexDirection: 'row',
  },
  questionView: {
    flex: 1,
    flexDirection: 'column',
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: 'orange',
  },
});
