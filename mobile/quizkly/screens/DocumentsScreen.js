import React from 'react';
import { ScrollView, StyleSheet, View } from 'react-native';
import { QuestionList } from '../components/Questions/QuestionList';

export default class DocumentsScreen extends React.Component {

  static navigationOptions = {
    title: 'Documents',
  };

  state = {
    list: [
      { key: '0', question: "What?", choices: [
        { key: '0', text: "ans1", color: 'green'},
        { key: '1', text: "ans2", color: 'red'},
        { key: '2', text: "ans3", color: 'red' },
      ] },
      { key: '1', question: "Really?", choices: [
        { key: '0', text: "ans1", color: 'red' },
        { key: '1', text: "ans2", color: 'green' },
        { key: '2', text: "ans3", color: 'red' },
      ] },
      { key: '2', question: "How?", choices: [
        { key: '0', text: "ans1", color: 'red' },
        { key: '1', text: "ans2", color: 'red' },
        { key: '2', text: "ans3", color: 'green' },
      ] },
    ],
  }

  render() {
    return (
      <View style={styles.container}>
        <QuestionList style={styles.questionList} list={this.state.list} />
      </View>
    );
  }
}

const styles = StyleSheet.create({
  questionList: {
    backgroundColor: 'red',
    flex: 1,
  },
  container: {
    flex: 1,
    paddingTop: 15,
    backgroundColor: '#ddd',
  },
});
